from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from app.attention_molds import AVAILABLE_MOLDS, MOLD_CATALOG
from app.llm import build_chain_llm
from create_mold_agent import create_mold_agent

router = APIRouter()

DEFAULT_PROMPT = """
You are MOLD Agent, an attention-respecting AI.
Use tools for actions and molds for cognitive shape.
For this run, call exactly one selected mold before the final answer.
After the mold returns, answer in the same minimal spirit.
Do not create extra options, drafts, summaries, or caveats unless the user asks.
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

    selected = [AVAILABLE_MOLDS[name] for name in body.mold_names if name in AVAILABLE_MOLDS]
    if not selected:
        raise HTTPException(status_code=400, detail="Select at least one mold")

    prompt = f"{body.system_prompt or DEFAULT_PROMPT}\n\nSelected molds: {', '.join(body.mold_names)}"
    agent = create_mold_agent(model=llm, tools=[], molds=selected, prompt=prompt)

    try:
        result = await agent.ainvoke({"messages": [HumanMessage(content=body.message)]}, config={"recursion_limit": 8})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
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
