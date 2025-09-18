"""
MOLD Agent - Modular Output Learning Design

A configurable agent framework that combines reasoning with structured output generation.
MOLD agents can dynamically create structured outputs using Pydantic models as "molds"
that shape raw conversation data into organized, typed formats.

Key Features:
- Universal mold tool generation from Pydantic models
- Smart routing based on tool name suffixes (_mold)
- Configurable tools, molds, models, and prompts
- Async-first architecture with beautiful message display
"""

from typing import Annotated, Dict, Any, Optional, Union
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command
import inspect
import json

# Global registry for dynamic mold state fields
_MOLD_STATE_REGISTRY = {}

# Global debug flag for controlling prints
_DEBUG_MODE = False

def mold(func):
    """Revolutionary mold decorator with auto-registration for dynamic MoldState"""
    # Extract field name from function name
    field_name = func.__name__.replace('_mold', '')

    # Get function signature to extract data type
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Find the data parameter (skip tool_call_id)
    data_param = None
    for param in params:
        if param.name != 'tool_call_id':
            data_param = param
            break

    if data_param and data_param.annotation != inspect.Parameter.empty:
        # Auto-register state field for dynamic MoldState
        _MOLD_STATE_REGISTRY[field_name] = data_param.annotation
        if _DEBUG_MODE:
            print(f"Mold '{field_name}' auto-registered with type: {data_param.annotation}")

    # Apply @tool decorator first
    decorated_tool = tool(func)

    # Override name to add _mold suffix automatically
    original_name = func.__name__
    decorated_tool.name = f"{original_name}_mold"

    # Enhance description to help LLM understand molds better
    original_desc = decorated_tool.description or 'Structure data into JSON format'
    decorated_tool.description = f"SCHEMA TOOL: {original_desc}. This tool helps you focus by structuring data into a defined schema. Use it to organize information and guide your data collection - the schema shows what information to look for when calling other tools."

    return decorated_tool





def create_dynamic_mold_state(molds):
    """Create MoldState dynamically based on registered molds"""
    # Base fields that every MoldState needs
    base_fields = {
        'messages': Annotated[list[BaseMessage], add_messages]
    }

    # Add fields from mold registry
    for mold_func in molds:
        # Extract field name from mold tool name
        field_name = mold_func.name.replace('_mold', '')

        if field_name in _MOLD_STATE_REGISTRY:
            field_type = _MOLD_STATE_REGISTRY[field_name]
            base_fields[field_name] = Optional[field_type]
            if _DEBUG_MODE:
                print(f"Added dynamic field '{field_name}': {field_type}")

    # Create TypedDict dynamically
    DynamicMoldState = TypedDict('DynamicMoldState', base_fields)
    return DynamicMoldState
    
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage"""
    
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

class BasicMoldNode:
    """A node that executes mold tool calls"""

    def __init__(self, mold_tools: list) -> None:
        self.mold_tools_by_name = {tool.name: tool for tool in mold_tools}

    def __call__(self, inputs: dict):
        message = inputs["messages"][-1]
        outputs = []
        state_updates = {}

        for tool_call in message.tool_calls:
            if tool_call["name"].endswith("_mold"):
                tool_result = self.mold_tools_by_name[tool_call["name"]].invoke(tool_call)

                # Handle Command return type
                if isinstance(tool_result, Command):
                    # Extract messages from Command
                    if "messages" in tool_result.update:
                        outputs.extend(tool_result.update["messages"])
                    # Extract state updates
                    for key, value in tool_result.update.items():
                        if key != "messages":
                            state_updates[key] = value
                else:
                    # Handle string return type (for non-Command molds)
                    outputs.append(
                        ToolMessage(
                            content=tool_result,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )

        result = {"messages": outputs}
        result.update(state_updates)
        return result

def smart_route(state):
    """Route to Tools/Molds/END"""
    ai_message = state["messages"][-1]

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        # Check if any tool call is a mold (has _mold suffix from decorator)
        for tool_call in ai_message.tool_calls:
            if tool_call["name"].endswith("_mold"):
                return "molds"
        return "tools"

    return END

def create_chatbot_node(model: Union[str, BaseChatModel], tools: list = None, molds: list = None, prompt: str = ""):
    """Create a configurable chatbot node"""

    async def chatbot(state) -> Dict[str, Any]:
        """🚀 Revolutionary MOLD Agent chatbot node with dynamic state compatibility"""
        # Use the model directly - same as create_react_agent pattern
        llm_model = model

        # Combine all tools (molds are already tools with @mold decorator)
        all_tools = tools + molds
        llm_with_tools = llm_model.bind_tools(all_tools)

        # Add system prompt if provided
        messages = state["messages"]
        if prompt and not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=prompt)] + messages

        response = await llm_with_tools.ainvoke(messages)

        return {"messages": [response]}

    return chatbot

def create_mold_agent(
    model: Union[str, BaseChatModel],
    tools: list = None,
    molds: list = None,
    prompt: str = "",
    debug: bool = False
) -> StateGraph:
    """🚀 Revolutionary MOLD Agent (Modular Output Learning Design) - The Future of create_react_agent"""

    # Set global debug mode
    global _DEBUG_MODE
    _DEBUG_MODE = debug

    # 🚀 Create dynamic state based on molds - REVOLUTIONARY!
    DynamicMoldState = create_dynamic_mold_state(molds or [])
    if _DEBUG_MODE:
        print(f"🚀 Created dynamic MoldState with fields: {list(DynamicMoldState.__annotations__.keys())}")

    # Create nodes
    chatbot_node = create_chatbot_node(model, tools, molds, prompt)
    tool_node = BasicToolNode(tools)
    mold_node = BasicMoldNode(molds)

    # Build the MOLD Agent graph with dynamic state
    graph_builder = StateGraph(DynamicMoldState)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("molds", mold_node)

    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        smart_route,
        {"tools": "tools", "molds": "molds", END: END},
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("molds", "chatbot")

    return graph_builder.compile()

# Export the MOLD Agent creation function
__all__ = ['create_mold_agent', 'mold']

