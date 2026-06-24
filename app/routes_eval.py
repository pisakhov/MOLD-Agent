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
    mold_name: str = "refined_minimal_final_response"


class VoteBody(BaseModel):
    criteria_votes: dict[str, str]


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


async def generate_rubric(llm, system_prompt: str, user_message: str, option_a: str, option_b: str):
    data = await ask_json(
        llm,
        "You generate comparison rubrics for blind AI-response voting. Output JSON only.",
        f"""
System prompt:
{system_prompt}

User message:
{user_message}

Option A:
{option_a}

Option B:
{option_b}

Generate 5 or 7 task-specific voting criteria.
The criteria must be unbiased between options and must not mention model, mold, no-tool, short, long, concise, verbose, Option A, or Option B.
Each criterion should ask about a meaningful quality a human can judge for this exact case.
Use an odd number so majority vote cannot tie.
Output exactly: {{"criteria":["criterion 1", "criterion 2", "..."]}}
""".strip(),
    )
    criteria = data.get("criteria", []) if isinstance(data, dict) else []
    criteria = [str(x).strip() for x in criteria if str(x).strip()]
    if len(criteria) >= 7:
        criteria = criteria[:7]
    elif len(criteria) >= 5:
        criteria = criteria[:5]
    elif len(criteria) >= 3:
        criteria = criteria[:3]
    if len(criteria) % 2 == 0:
        criteria = criteria[:-1]
    if len(criteria) < 3:
        raise ValueError("Rubric generator did not produce enough criteria")
    return criteria


def history_for_judge():
    examples = []
    for item in eval_store.voted_examples():
        examples.append({
            "system_prompt": item["system_prompt"],
            "user_message": item["user_message"],
            "rubric": json.loads(item["rubric_json"] or "[]"),
            "option_a": item["option_a_answer"],
            "option_b": item["option_b_answer"],
            "criteria_votes": json.loads(item["criteria_votes_json"] or "{}"),
            "human_winner": item["human_winner"],
        })
    return examples


async def suggest_judge(llm, system_prompt: str, user_message: str, option_a: str, option_b: str, rubric: list[str]):
    data = await ask_json(
        llm,
        "You predict a human's blind A/B preference. You do not decide for them. Output JSON only.",
        f"""
Past human votes, newest first:
{json.dumps(history_for_judge(), ensure_ascii=False, indent=2)}

Current system prompt:
{system_prompt}

Current user message:
{user_message}

Rubric visible to the human:
{json.dumps(rubric, ensure_ascii=False, indent=2)}

Option A:
{option_a}

Option B:
{option_b}

Predict which option this human is more likely to choose after voting by criteria.
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
    out = {
        "id": case["id"],
        "batch_id": case["batch_id"],
        "system_prompt": case["system_prompt"],
        "user_message": case["user_message"],
        "mold_name": case["mold_name"],
        "option_a_answer": case["option_a_answer"],
        "option_b_answer": case["option_b_answer"],
        "rubric": json.loads(case["rubric_json"] or "[]"),
        "voted": voted,
    }
    if reveal:
        out.update({
            "option_a_kind": case["option_a_kind"],
            "option_b_kind": case["option_b_kind"],
            "human_winner": case.get("human_winner"),
            "criteria_votes": json.loads(case.get("criteria_votes_json") or "{}"),
            "judge_suggestion": case.get("judge_suggestion"),
            "judge_confidence": case.get("judge_confidence"),
            "judge_rationale": case.get("judge_rationale"),
            "judge_agreed": case.get("human_winner") and case.get("judge_suggestion") == case.get("human_winner"),
            "mold_state": json.loads(case["mold_state_json"] or "{}"),
            "mold_trace": json.loads(case["mold_trace_json"] or "[]"),
        })
    return out


async def build_case(batch_id: str, mold_name: str, generator_llm):
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

    rubric = await generate_rubric(build_chain_llm(), pair["system_prompt"], pair["user_message"], option_a_answer, option_b_answer)
    judge = await suggest_judge(build_chain_llm(), pair["system_prompt"], pair["user_message"], option_a_answer, option_b_answer, rubric)

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
        "rubric": rubric,
        "judge_suggestion": judge["winner"],
        "judge_confidence": judge["confidence"],
        "judge_rationale": judge["rationale"],
    })


@router.post("/batch")
async def create_batch(body: BatchCreate):
    if body.mold_name not in AVAILABLE_MOLDS:
        raise HTTPException(status_code=400, detail="Unknown mold")
    generator_llm = build_chain_llm()
    if not generator_llm:
        raise HTTPException(status_code=400, detail="Configure at least one model in Settings first")

    await ensure_samples(generator_llm)
    if eval_store.sample_count() < BATCH_SIZE:
        raise HTTPException(status_code=400, detail="Could not bootstrap enough eval samples; try again")
    batch_id = eval_store.create_batch(body.mold_name, BATCH_SIZE)
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

    rubric = json.loads(case["rubric_json"] or "[]")
    votes = {str(k): str(v).upper()[:1] for k, v in body.criteria_votes.items()}
    if len(votes) != len(rubric) or any(votes.get(str(i)) not in ("A", "B") for i in range(len(rubric))):
        raise HTTPException(status_code=400, detail="Vote on every criterion")
    a_votes = sum(1 for v in votes.values() if v == "A")
    winner = "A" if a_votes > len(rubric) / 2 else "B"
    updated = eval_store.vote_case(case_id, winner, votes)
    return {"case": public_case(updated, reveal=True)}


@router.get("/stats")
def eval_stats():
    data = eval_store.stats()
    summary = data["summary"] or {}
    voted = summary.get("voted_cases") or 0
    judge_correct = summary.get("judge_correct") or 0
    mold_wins = 0
    for item in data["per_mold"]:
        item["mold_win_rate"] = (item["mold_wins"] or 0) / item["voted"] if item["voted"] else 0
        item["judge_accuracy"] = (item["judge_correct"] or 0) / item["voted"] if item["voted"] else 0
        mold_wins += item["mold_wins"] or 0
    summary["mold_wins"] = mold_wins
    summary["mold_win_rate"] = mold_wins / voted if voted else 0
    summary["judge_accuracy"] = judge_correct / voted if voted else 0
    return data


@router.get("/export")
def export_jsonl():
    lines = []
    for case in eval_store.list_cases(pending=False, limit=10000):
        lines.append(json.dumps(public_case(case, reveal=True), ensure_ascii=False))
    return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""), media_type="application/jsonl")
