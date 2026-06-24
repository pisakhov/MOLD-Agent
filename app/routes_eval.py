import asyncio
import json
import random
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app import eval_store
from app.attention_molds import AVAILABLE_MOLDS
from app.llm import build_chain_llm
from app.routes_agent import DEFAULT_PROMPT, RunRequest, extract_text, run_mold_arm, run_simple_arm

router = APIRouter()
BATCH_SIZE = 5


class BatchCreate(BaseModel):
    mold_name: str | None = None


class VoteBody(BaseModel):
    criteria_votes: dict[str, str] | None = None
    feedback_votes: dict[str, str] | None = None


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    starts = [i for i in [text.find("{"), text.find("[")] if i >= 0]
    if not starts:
        raise ValueError("LLM did not return JSON")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        raise ValueError("LLM returned incomplete JSON")
    return json.loads(text[start:end + 1])


async def ask_json(llm, system: str, user: str):
    response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    return extract_json(extract_text(response.content))


async def ensure_samples(llm):
    missing = max(0, BATCH_SIZE - eval_store.sample_count())
    if not missing:
        return
    data = await ask_json(
        llm,
        "You generate seed samples for a blind AI-response preference lab. Output JSON only.",
        f"""
Generate {missing} highly diverse pairs of system_prompt and user_message.
The system prompt and user message may intentionally mismatch.
Do not explain. Output exactly:
{{"pairs":[{{"system_prompt":"...","user_message":"..."}}]}}
""".strip(),
    )
    pairs = data.get("pairs", []) if isinstance(data, dict) else []
    for pair in pairs[:missing]:
        if pair.get("system_prompt") and pair.get("user_message"):
            eval_store.create_sample(pair["system_prompt"], pair["user_message"], "seed")


async def generate_pair(llm, samples: list[dict[str, str]]):
    data = await ask_json(
        llm,
        "You generate test cases for a blind AI-response preference lab. Output JSON only.",
        f"""
Here are five existing sample pairs. Treat them as anti-examples:
{json.dumps(samples, ensure_ascii=False, indent=2)}

Generate one new pair that does not resemble any of them in topic, intent, tone, format, domain, or relationship between system prompt and user message.
It is allowed and useful for the system prompt and user message to be mismatched.
No explanation outside JSON. Output exactly:
{{"system_prompt":"...","user_message":"...","novelty_note":"why this is unlike the samples"}}
""".strip(),
    )
    if not data.get("system_prompt") or not data.get("user_message"):
        raise ValueError("Generated pair is missing system_prompt or user_message")
    return data


def clamp_signal(value: Any):
    try:
        return max(-1.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def normalize_feedback_items(raw: Any):
    if isinstance(raw, dict):
        raw_items = raw.get("items", raw.get("criteria", []))
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raw_items = []

    items = []
    for index, item in enumerate(raw_items):
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            items.append({
                "id": f"q{index + 1}",
                "kind": "comparison",
                "focus": "legacy criterion",
                "question": text,
                "target": "both",
                "anchor_option": None,
                "anchor_text": "",
                "why_ask": "Imported from the older rubric format.",
                "choices": [
                    {"id": "A", "label": "Option A", "signals": {"A": 1, "B": 0}},
                    {"id": "B", "label": "Option B", "signals": {"A": 0, "B": 1}},
                ],
            })
            continue
        if not isinstance(item, dict):
            continue

        question = str(item.get("question") or item.get("prompt") or "").strip()
        if not question:
            continue

        choices = []
        seen_choice_ids = set()
        for choice_index, choice in enumerate(item.get("choices", [])):
            if not isinstance(choice, dict):
                continue
            label = str(choice.get("label") or choice.get("text") or "").strip()
            if not label:
                continue
            choice_id = str(choice.get("id") or f"c{choice_index + 1}").strip()
            if not choice_id or choice_id in seen_choice_ids:
                choice_id = f"c{choice_index + 1}"
            seen_choice_ids.add(choice_id)
            signals = choice.get("signals") if isinstance(choice.get("signals"), dict) else {}
            choices.append({
                "id": choice_id,
                "label": label,
                "signals": {
                    "A": clamp_signal(signals.get("A")),
                    "B": clamp_signal(signals.get("B")),
                },
            })

        if len(choices) < 2:
            continue
        if not any(choice["signals"]["A"] or choice["signals"]["B"] for choice in choices):
            continue

        anchor_option = str(item.get("anchor_option") or "").upper()[:1]
        if anchor_option not in ("A", "B"):
            anchor_option = None
        item_id = str(item.get("id") or f"q{index + 1}").strip() or f"q{index + 1}"
        if any(existing["id"] == item_id for existing in items):
            item_id = f"q{index + 1}"
        items.append({
            "id": item_id,
            "kind": str(item.get("kind") or ("annotation" if anchor_option else "comparison")).strip() or "comparison",
            "focus": str(item.get("focus") or "").strip(),
            "question": question,
            "target": str(item.get("target") or (anchor_option or "both")).strip() or "both",
            "anchor_option": anchor_option,
            "anchor_text": str(item.get("anchor_text") or "").strip(),
            "why_ask": str(item.get("why_ask") or item.get("rationale") or "").strip(),
            "choices": choices[:4],
        })
    return items[:8]


async def generate_feedback_items(llm, system_prompt: str, user_message: str, option_a: str, option_b: str):
    data = await ask_json(
        llm,
        "You generate precise human feedback prompts for blind AI-response evaluation. Output JSON only.",
        f"""
System prompt:
{system_prompt}

User message:
{user_message}

Option A:
{option_a}

Option B:
{option_b}

Read the actual wording of both options and generate 5 to 8 feedback items about how each response responds.
Do not grade whether the answer's facts are correct, whether it solved the user's task, or which answer you generally prefer.
Ask about observable response behavior: judgment, concision, depth, originality, specificity, framing, care, tone, or other qualities you notice in these exact outputs.
Those qualities are examples, not a checklist. The items must be discovered from the outputs, not copied from a generic rubric.

Include a natural mix of:
- comparison items where a human chooses which option better demonstrates a response quality;
- annotation items tied to an exact sentence or short phrase from one option, asking whether that wording is a good or bad signal.

Rules:
- Every question and every choice label must be specific to this case.
- No reusable rubric wording, placeholders, or templates.
- Do not mention hidden implementation details, models, tools, molds, or no-tool agents.
- For annotation items, anchor_option must be "A" or "B" and anchor_text must be an exact substring copied from that option, preferably 3 to 25 words.
- For each choice, include numeric signals with keys A and B. Positive means a good signal for that option; negative means a bad signal. Use 0 for unaffected options.
- Use compact choice labels that describe the human feedback, not just "yes" or "no" unless those words are genuinely enough.

Output exactly this JSON shape:
{{
  "items": [
    {{
      "id": "q1",
      "kind": "comparison or annotation",
      "focus": "short human-readable quality being tested",
      "question": "specific feedback question",
      "target": "both, A, or B",
      "anchor_option": null,
      "anchor_text": "",
      "why_ask": "why this is a useful signal",
      "choices": [
        {{"id": "choice-id", "label": "generated choice label", "signals": {{"A": 1, "B": 0}}}}
      ]
    }}
  ]
}}
""".strip(),
    )
    items = normalize_feedback_items(data)
    if len(items) < 3:
        raise ValueError("Feedback generator did not produce enough usable items")
    return items


def decode_vote_record(raw: str | None):
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        data = {}
    if isinstance(data, dict) and isinstance(data.get("votes"), dict):
        return {
            "version": data.get("version") or "feedback-v1",
            "votes": {str(k): str(v) for k, v in data.get("votes", {}).items()},
            "scores": data.get("scores") if isinstance(data.get("scores"), dict) else {},
        }
    if isinstance(data, dict):
        return {"version": "legacy", "votes": {str(k): str(v) for k, v in data.items()}, "scores": {}}
    return {"version": "legacy", "votes": {}, "scores": {}}


def score_feedback(items: list[dict[str, Any]], votes: dict[str, str]):
    scores = {"A": 0.0, "B": 0.0}
    normalized_votes = {str(k): str(v) for k, v in votes.items()}
    for index, item in enumerate(items):
        item_id = str(item.get("id") or f"q{index + 1}")
        choice_id = normalized_votes.get(item_id, normalized_votes.get(str(index)))
        if choice_id is None:
            raise ValueError("Vote on every feedback item")
        choices = item.get("choices") if isinstance(item.get("choices"), list) else []
        selected = next((choice for choice in choices if str(choice.get("id")) == choice_id), None)
        if not selected:
            raise ValueError("Unknown feedback choice")
        signals = selected.get("signals") if isinstance(selected.get("signals"), dict) else {}
        scores["A"] += clamp_signal(signals.get("A"))
        scores["B"] += clamp_signal(signals.get("B"))
    return scores


def history_for_judge():
    examples = []
    for item in eval_store.voted_examples():
        vote_record = decode_vote_record(item.get("criteria_votes_json"))
        examples.append({
            "system_prompt": item["system_prompt"],
            "user_message": item["user_message"],
            "feedback_items": normalize_feedback_items(json.loads(item["rubric_json"] or "[]")),
            "option_a": item["option_a_answer"],
            "option_b": item["option_b_answer"],
            "feedback_votes": vote_record["votes"],
            "feedback_scores": vote_record["scores"],
            "signal_winner": item["human_winner"],
        })
    return examples


async def suggest_judge(llm, system_prompt: str, user_message: str, option_a: str, option_b: str, feedback_items: list[dict[str, Any]]):
    data = await ask_json(
        llm,
        "You predict how a human will score blind AI-response feedback prompts. You do not decide for them. Output JSON only.",
        f"""
Past human feedback, newest first:
{json.dumps(history_for_judge(), ensure_ascii=False, indent=2)}

Current system prompt:
{system_prompt}

Current user message:
{user_message}

Feedback prompts visible to the human:
{json.dumps(feedback_items, ensure_ascii=False, indent=2)}

Option A:
{option_a}

Option B:
{option_b}

Predict which option is more likely to receive the stronger net human feedback signal after these prompts are answered.
Do not mention hidden implementation details. Output exactly:
{{"suggested_winner":"A or B","confidence":0.0,"rationale":"short prediction rationale"}}
""".strip(),
    )
    winner = str(data.get("suggested_winner", "")).upper()[:1]
    if winner not in ("A", "B"):
        winner = "A"
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return {"winner": winner, "confidence": max(0, min(1, confidence)), "rationale": str(data.get("rationale", ""))}


def public_case(case: dict[str, Any] | None, reveal: bool = False):
    if not case:
        return None
    voted = bool(case.get("human_winner"))
    reveal = reveal or voted
    feedback_items = normalize_feedback_items(json.loads(case["rubric_json"] or "[]"))
    out = {
        "id": case["id"],
        "batch_id": case["batch_id"],
        "system_prompt": case["system_prompt"],
        "user_message": case["user_message"],
        "mold_name": case["mold_name"],
        "option_a_answer": case["option_a_answer"],
        "option_b_answer": case["option_b_answer"],
        "feedback_items": feedback_items,
        "rubric": feedback_items,
        "voted": voted,
    }
    if reveal:
        vote_record = decode_vote_record(case.get("criteria_votes_json"))
        scores = vote_record.get("scores") or {}
        if not scores and vote_record["votes"]:
            try:
                scores = score_feedback(feedback_items, vote_record["votes"])
            except ValueError:
                scores = {}
        out.update({
            "option_a_kind": case["option_a_kind"],
            "option_b_kind": case["option_b_kind"],
            "human_winner": case.get("human_winner"),
            "feedback_votes": vote_record["votes"],
            "feedback_scores": scores,
            "criteria_votes": vote_record["votes"],
            "judge_suggestion": case.get("judge_suggestion"),
            "judge_confidence": case.get("judge_confidence"),
            "judge_rationale": case.get("judge_rationale"),
            "judge_agreed": case.get("human_winner") in ("A", "B") and case.get("judge_suggestion") == case.get("human_winner"),
            "mold_state": json.loads(case["mold_state_json"] or "{}"),
            "mold_trace": json.loads(case["mold_trace_json"] or "[]"),
        })
    return out


async def build_case(batch_id: str, requested_mold_name: str | None, generator_llm):
    mold_name = requested_mold_name or random.choice(list(AVAILABLE_MOLDS))
    samples = eval_store.random_samples(BATCH_SIZE)
    pair = await generate_pair(generator_llm, samples)
    eval_store.create_sample(pair["system_prompt"], pair["user_message"], "case")

    request = RunRequest(message=pair["user_message"], mold_names=[mold_name], system_prompt=pair["system_prompt"])
    simple_task = run_simple_arm(request, build_chain_llm())
    mold_task = run_mold_arm(request, build_chain_llm())
    simple, mold = await asyncio.gather(simple_task, mold_task)

    if random.random() < 0.5:
        option_a_kind, option_b_kind = "simple", "mold"
        option_a_answer, option_b_answer = simple["answer"], mold["answer"]
    else:
        option_a_kind, option_b_kind = "mold", "simple"
        option_a_answer, option_b_answer = mold["answer"], simple["answer"]

    feedback_items = await generate_feedback_items(build_chain_llm(), pair["system_prompt"], pair["user_message"], option_a_answer, option_b_answer)
    judge = await suggest_judge(build_chain_llm(), pair["system_prompt"], pair["user_message"], option_a_answer, option_b_answer, feedback_items)

    return eval_store.create_case({
        "batch_id": batch_id,
        "seed_samples": samples,
        "system_prompt": pair["system_prompt"],
        "user_message": pair["user_message"],
        "generation_notes": pair.get("novelty_note", ""),
        "mold_name": mold_name,
        "option_a_kind": option_a_kind,
        "option_b_kind": option_b_kind,
        "option_a_answer": option_a_answer,
        "option_b_answer": option_b_answer,
        "simple_answer": simple["answer"],
        "mold_answer": mold["answer"],
        "mold_state": mold.get("mold_state", {}),
        "mold_trace": mold.get("trace", []),
        "rubric": feedback_items,
        "judge_suggestion": judge["winner"],
        "judge_confidence": judge["confidence"],
        "judge_rationale": judge["rationale"],
    })


@router.post("/batch")
async def create_batch(body: BatchCreate):
    if body.mold_name and body.mold_name not in AVAILABLE_MOLDS:
        raise HTTPException(status_code=400, detail="Unknown mold")
    generator_llm = build_chain_llm()
    if not generator_llm:
        raise HTTPException(status_code=400, detail="Configure at least one model in Settings first")

    await ensure_samples(generator_llm)
    if eval_store.sample_count() < BATCH_SIZE:
        raise HTTPException(status_code=400, detail="Could not bootstrap enough eval samples; try again")
    batch_id = eval_store.create_batch(body.mold_name or "auto", BATCH_SIZE)
    cases = []
    try:
        for _ in range(BATCH_SIZE):
            cases.append(await build_case(batch_id, body.mold_name, generator_llm))
        eval_store.finish_batch(batch_id)
    except Exception as exc:
        eval_store.finish_batch(batch_id, "failed", str(exc))
        raise HTTPException(status_code=400, detail=str(exc))
    return {"batch_id": batch_id, "cases": [public_case(case) for case in cases]}


@router.get("/next")
def next_eval_case():
    return {"case": public_case(eval_store.next_case())}


@router.get("/cases")
def list_eval_cases(pending: bool | None = None):
    return {"cases": [public_case(case, reveal=True) for case in eval_store.list_cases(pending=pending)]}


@router.post("/cases/{case_id}/vote")
def vote(case_id: str, body: VoteBody):
    case = eval_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Eval case not found")
    if case.get("human_winner"):
        return {"case": public_case(case, reveal=True)}

    feedback_items = normalize_feedback_items(json.loads(case["rubric_json"] or "[]"))
    raw_votes = body.feedback_votes if body.feedback_votes is not None else body.criteria_votes
    votes = {str(k): str(v) for k, v in (raw_votes or {}).items()}
    canonical_votes = {}
    for index, item in enumerate(feedback_items):
        item_id = str(item.get("id") or f"q{index + 1}")
        choice_id = votes.get(item_id, votes.get(str(index)))
        if choice_id is not None:
            canonical_votes[item_id] = choice_id
    if len(canonical_votes) != len(feedback_items):
        raise HTTPException(status_code=400, detail="Vote on every feedback item")
    try:
        scores = score_feedback(feedback_items, canonical_votes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if scores["A"] == scores["B"]:
        winner = "tie"
    else:
        winner = "A" if scores["A"] > scores["B"] else "B"
    updated = eval_store.vote_case(case_id, winner, {"version": "feedback-v1", "votes": canonical_votes, "scores": scores})
    return {"case": public_case(updated, reveal=True)}


def feedback_signal_stats():
    summary = {
        "signal_cases": 0,
        "mold_signal_score": 0.0,
        "simple_signal_score": 0.0,
        "mold_signal_avg": 0.0,
        "simple_signal_avg": 0.0,
        "mold_signal_delta": 0.0,
    }
    per_mold: dict[str, dict[str, float]] = {}
    for case in eval_store.list_cases(pending=False, limit=10000):
        vote_record = decode_vote_record(case.get("criteria_votes_json"))
        scores = vote_record.get("scores") or {}
        if not scores and vote_record["votes"]:
            try:
                scores = score_feedback(normalize_feedback_items(json.loads(case["rubric_json"] or "[]")), vote_record["votes"])
            except ValueError:
                scores = {}
        if not scores:
            continue
        a_score = float(scores.get("A") or 0)
        b_score = float(scores.get("B") or 0)
        mold_score = a_score if case["option_a_kind"] == "mold" else b_score
        simple_score = a_score if case["option_a_kind"] == "simple" else b_score
        summary["signal_cases"] += 1
        summary["mold_signal_score"] += mold_score
        summary["simple_signal_score"] += simple_score

        mold_name = case["mold_name"]
        bucket = per_mold.setdefault(mold_name, {
            "signal_cases": 0,
            "mold_signal_score": 0.0,
            "simple_signal_score": 0.0,
            "mold_signal_avg": 0.0,
            "simple_signal_avg": 0.0,
            "mold_signal_delta": 0.0,
        })
        bucket["signal_cases"] += 1
        bucket["mold_signal_score"] += mold_score
        bucket["simple_signal_score"] += simple_score

    if summary["signal_cases"]:
        summary["mold_signal_avg"] = summary["mold_signal_score"] / summary["signal_cases"]
        summary["simple_signal_avg"] = summary["simple_signal_score"] / summary["signal_cases"]
        summary["mold_signal_delta"] = summary["mold_signal_avg"] - summary["simple_signal_avg"]
    for bucket in per_mold.values():
        if bucket["signal_cases"]:
            bucket["mold_signal_avg"] = bucket["mold_signal_score"] / bucket["signal_cases"]
            bucket["simple_signal_avg"] = bucket["simple_signal_score"] / bucket["signal_cases"]
            bucket["mold_signal_delta"] = bucket["mold_signal_avg"] - bucket["simple_signal_avg"]
    return {"summary": summary, "per_mold": per_mold}


@router.get("/stats")
def eval_stats():
    data = eval_store.stats()
    summary = data["summary"] or {}
    voted = summary.get("voted_cases") or 0
    judge_correct = summary.get("judge_correct") or 0
    mold_wins = 0
    signals = feedback_signal_stats()
    for item in data["per_mold"]:
        item["mold_win_rate"] = (item["mold_wins"] or 0) / item["voted"] if item["voted"] else 0
        item["judge_accuracy"] = (item["judge_correct"] or 0) / item["voted"] if item["voted"] else 0
        item.update(signals["per_mold"].get(item["mold_name"], {
            "signal_cases": 0,
            "mold_signal_score": 0.0,
            "simple_signal_score": 0.0,
            "mold_signal_avg": 0.0,
            "simple_signal_avg": 0.0,
            "mold_signal_delta": 0.0,
        }))
        mold_wins += item["mold_wins"] or 0
    summary["mold_wins"] = mold_wins
    summary["mold_win_rate"] = mold_wins / voted if voted else 0
    summary["judge_accuracy"] = judge_correct / voted if voted else 0
    summary.update(signals["summary"])
    return data


@router.get("/export")
def export_jsonl():
    lines = []
    for case in eval_store.list_cases(pending=False, limit=10000):
        lines.append(json.dumps(public_case(case, reveal=True), ensure_ascii=False))
    return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""), media_type="application/jsonl")
