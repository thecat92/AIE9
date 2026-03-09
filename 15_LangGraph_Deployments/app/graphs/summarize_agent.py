"""An agent graph with a post-response summarizer.

After the agent responds, a secondary node summarize the response.
If summary is clear, end; otherwise, continue the loop or terminate after a safe limit.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from app.state import MessagesState
from app.models import get_chat_model
from app.tools import get_tool_belt


class SummarizationResult(BaseModel):
    summary: str = Field(description="A concise summary of the response based on the initial query")


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: MessagesState) -> dict:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def route_to_action_or_summarization(state: MessagesState):
    """Decide whether to execute tools or run the summarization evaluator."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "summarization"


_summarization_prompt = ChatPromptTemplate.from_template(
    "Given an initial query and a final response, generate a concise summary of the response based on the initial query.\n\n"
    "The summary should be concise and to the point, and should be based on the initial query.\n\n"
    "Initial Query:\n{initial_query}\n\n"
    "Final Response:\n{final_response}"
)


def summarization_node(state: MessagesState) -> dict:
    """Generate a concise summary of the latest response based on the initial query."""
    if len(state["messages"]) > 10:
        return {"messages": [AIMessage(content="SUMMARIZATION:END")]}

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]

    structured_model = get_chat_model(model_name="gpt-4.1-mini").with_structured_output(SummarizationResult)
    result = (_summarization_prompt | structured_model).invoke(
        {
            "initial_query": initial_query.content,
            "final_response": final_response.content,
        }
    )

    decision = "Y" if result.summary is not None else "N"
    return {"messages": [AIMessage(content=f"SUMMARIZATION:{decision}")]}


def summarization_decision(state: MessagesState):
    """Terminate on 'SUMMARIZATION:Y' or loop otherwise; guard against infinite loops."""
    if any(getattr(m, "content", "") == "SUMMARIZATION:END" for m in state["messages"][-1:]):
        return END

    last = state["messages"][-1]
    text = getattr(last, "content", "")
    if "SUMMARIZATION:Y" in text:
        return "end"
    return "continue"


def build_graph():
    """Build an agent graph with an auxiliary summarization evaluation subgraph."""
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("summarization", summarization_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_summarization,
        {"action": "action", "summarization": "summarization"},
    )
    graph.add_conditional_edges(
        "summarization",
        summarization_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    return graph


graph = build_graph().compile()
