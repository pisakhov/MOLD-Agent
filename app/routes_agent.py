import asyncio

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from app.attention_molds import AVAILABLE_MOLDS, MOLD_CATALOG
from app.llm import build_chain_llm
from create_mold_agent import create_mold_agent

router = APIRouter()

DEFAULT_PROMPT = """
You are an attention-respecting AI.
Answer clearly and minimally.
Do not create extra options, drafts, summaries, or caveats unless the user asks.
""".strip()

MOLD_CONTRACT = """
Use the selected mold as a cognitive shape.
For this run, call exactly one selected mold before the final answer.
After the mold returns, answer in the same minimal spirit.
""".strip()


class RunRequest(BaseModel):
    message: str
    mold_names: list[str] = ["refined_minimal_final_response"]
    system_prompt: str | None = None


@router.get("/molds")
def molds():
    return {"molds": MOLD_CATALOG}


@router.post("/run")
async def run_agent(body: RunRequest):
    llm = build_chain_llm()
    if not llm:
        raise HTTPException(status_code=400, detail="Configure at least one model in Settings first")
    try:
        return await run_mold_arm(body, llm)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/compare")
async def compare_agents(body: RunRequest):
    simple_llm = build_chain_llm()
    mold_llm = build_chain_llm()
    if not simple_llm or not mold_llm:
        raise HTTPException(status_code=400, detail="Configure at least one model in Settings first")

    async def capture(label, fn):
        try:
            return await fn()
        except Exception as exc:
            return {"label": label, "answer": "", "error": str(exc)}

    simple, mold = await asyncio.gather(
        capture("Simple agent", lambda: run_simple_arm(body, simple_llm)),
        capture("Mold agent", lambda: run_mold_arm(body, mold_llm)),
    )
    return {"simple": simple, "mold": mold, "system_prompt": body.system_prompt or DEFAULT_PROMPT}


async def run_simple_arm(body: RunRequest, llm):
    prompt = body.system_prompt or DEFAULT_PROMPT
    messages = [SystemMessage(content=prompt), HumanMessage(content=body.message)]
    response = await llm.ainvoke(messages)
    return {"label": "Simple agent", "answer": extract_text(response.content), "trace": []}


async def run_mold_arm(body: RunRequest, llm):
    selected = [AVAILABLE_MOLDS[name] for name in body.mold_names if name in AVAILABLE_MOLDS]
    if not selected:
        raise ValueError("Select at least one mold")
    base_prompt = body.system_prompt or DEFAULT_PROMPT
    prompt = f"{base_prompt}\n\n{MOLD_CONTRACT}\nSelected molds: {', '.join(body.mold_names)}"
    agent = create_mold_agent(model=llm, tools=[], molds=selected, prompt=prompt)
    result = await agent.ainvoke({"messages": [HumanMessage(content=body.message)]}, config={"recursion_limit": 8})
    return {
        "label": "Mold agent",
        "answer": final_answer(result["messages"]),
        "mold_state": {name: result.get(name) for name in body.mold_names if result.get(name) is not None},
        "trace": trace_messages(result["messages"]),
    }


def final_answer(messages):
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return extract_text(msg.content)
    return ""


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
    return str(content)


def trace_messages(messages):
    trace = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            trace.append({"type": "ai_tool_calls", "tool_calls": msg.tool_calls})
        elif isinstance(msg, ToolMessage):
            trace.append({"type": "tool", "name": msg.name, "content": msg.content})
    return trace
