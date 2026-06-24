import json
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field

from create_mold_agent import mold


class MinimalFinalResponse(BaseModel):
    one_line_answer: str = Field(description="The shortest useful answer.")
    short_explanation: str = Field(description="One or two sentences. No bloat.")
    expand_if_wanted: str = Field(description="A small invitation to expand only if useful.")


class DigestNotFeed(BaseModel):
    matters: list[str] = Field(description="Only the important items.")
    can_ignore: list[str] = Field(description="Items the human does not need to process now.")
    next_action: str = Field(description="One concrete next action.")


class OneDecisionOnly(BaseModel):
    decision: str = Field(description="The one decision the human needs to make.")
    recommendation: str = Field(description="The recommended choice.")
    why: str = Field(description="Short reason, max two sentences.")


class DeleteInsteadOfGenerate(BaseModel):
    delete_or_skip: list[str] = Field(description="Outputs, steps, or explanations that should not be created.")
    keep: list[str] = Field(description="Only the work that remains valuable.")
    result: str = Field(description="The minimal useful result.")


class CloseLoopNotDraft(BaseModel):
    completed_result: str = Field(description="The finished answer/action, not a draft pile.")
    remaining_human_input: str = Field(description="Only what truly needs the human, or 'none'.")


class AttentionBudgetCheck(BaseModel):
    attention_cost: str = Field(description="tiny, small, medium, or high.")
    worth_it: bool = Field(description="Whether this deserves human attention now.")
    compressed_response: str = Field(description="The smallest useful response.")


def payload(data):
    return data.model_dump() if hasattr(data, "model_dump") else dict(data)


def message(name: str, data: dict, tool_call_id: str):
    return ToolMessage(json.dumps({name: data}, indent=2), tool_call_id=tool_call_id)


@mold
def refined_minimal_final_response(response: MinimalFinalResponse, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Use this for final answers that respect attention: one-line answer, short explanation, expand only on request. Default small."""
    data = payload(response)
    return Command(update={"refined_minimal_final_response": data, "messages": [message("refined_minimal_final_response", data, tool_call_id)]})


@mold
def digest_not_feed(digest: DigestNotFeed, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Use this when the user needs compression, not another feed. Keep only what matters, what can be ignored, and one next action."""
    data = payload(digest)
    return Command(update={"digest_not_feed": data, "messages": [message("digest_not_feed", data, tool_call_id)]})


@mold
def one_decision_only(packet: OneDecisionOnly, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Use this when the answer should reduce choices to one decision, one recommendation, and a short why."""
    data = payload(packet)
    return Command(update={"one_decision_only": data, "messages": [message("one_decision_only", data, tool_call_id)]})


@mold
def delete_instead_of_generate(plan: DeleteInsteadOfGenerate, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Use this when restraint matters. Delete or skip unnecessary outputs before producing the minimal useful result."""
    data = payload(plan)
    return Command(update={"delete_instead_of_generate": data, "messages": [message("delete_instead_of_generate", data, tool_call_id)]})


@mold
def close_loop_not_draft(loop: CloseLoopNotDraft, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Use this when the agent should finish a small loop instead of handing the human another draft to manage."""
    data = payload(loop)
    return Command(update={"close_loop_not_draft": data, "messages": [message("close_loop_not_draft", data, tool_call_id)]})


@mold
def attention_budget_check(check: AttentionBudgetCheck, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Use this before expanding. Estimate attention cost, decide if it is worth attention now, then compress."""
    data = payload(check)
    return Command(update={"attention_budget_check": data, "messages": [message("attention_budget_check", data, tool_call_id)]})


MOLD_CATALOG = [
    {"name": "refined_minimal_final_response", "title": "Minimal final response", "description": "Layered answer: one line, short reason, expand only if asked."},
    {"name": "digest_not_feed", "title": "Digest, not feed", "description": "Collapse many inputs into matters / ignore / next action."},
    {"name": "one_decision_only", "title": "One decision only", "description": "Reduce options into one human decision and recommendation."},
    {"name": "delete_instead_of_generate", "title": "Delete instead of generate", "description": "Remove low-value output before answering."},
    {"name": "close_loop_not_draft", "title": "Close loop, not draft", "description": "Finish small loops instead of creating review piles."},
    {"name": "attention_budget_check", "title": "Attention budget check", "description": "Compress based on attention cost."},
]

AVAILABLE_MOLDS = {
    "refined_minimal_final_response": refined_minimal_final_response,
    "digest_not_feed": digest_not_feed,
    "one_decision_only": one_decision_only,
    "delete_instead_of_generate": delete_instead_of_generate,
    "close_loop_not_draft": close_loop_not_draft,
    "attention_budget_check": attention_budget_check,
}
