from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from backend.utils import setup_logging, get_env_var
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from backend.tools import vector_search_tool, web_search_tool, pdf_generator_tool

logger = setup_logging(__name__)

class AgentState(TypedDict):
    """Represents the state passed between LangGraph nodes."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    generate_pdf: bool

def research_agent_node(state: AgentState):
    """ReAct agent to reason, query vector DB, and search the web."""
    logger.info("Entering research_agent_node")
    try:
        llm = ChatOpenAI(temperature=0, api_key=get_env_var("LLM_API_KEY"))
        tools = [vector_search_tool, web_search_tool]
        llm_with_tools = llm.bind_tools(tools)
        
        messages = state['messages']
        # Inject system prompt if it's the first execution
        if len(messages) == 1:
            sys_msg = SystemMessage(
                content="You are a research assistant. Use vector search for uploaded documents and web search for external info. Combine both contexts to form a comprehensive answer."
            )
            messages = [sys_msg] + messages
            
        response = llm_with_tools.invoke(messages)
        logger.info("Research agent reasoning step complete.")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Research agent node failed: {str(e)}")
        return {"messages": [AIMessage(content="Sorry, an error occurred during research.")]} 

def should_continue(state: AgentState) -> Literal["tools", "orchestrator_agent"]:
    """Determines whether to execute tools or move to the orchestrator."""
    messages = state['messages']
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tools"
    return "orchestrator_agent"

def orchestrator_node(state: AgentState):
    """Decides if the final answer should be streamed or returned as a PDF."""
    logger.info("Entering orchestrator_node")
    try:
        messages = state['messages']
        final_answer = messages[-1].content
        
        if state.get("generate_pdf"):
            logger.info("User requested notes PDF, triggering format.")
            # We explicitly invoke the tool here or create a final message regarding the PDF
            pdf_path = pdf_generator_tool.invoke(final_answer)
            pdf_msg = AIMessage(content=f"Research complete. Notes PDF generated at: {pdf_path}")
            return {"messages": [pdf_msg]}
        
        logger.info("Response finalized for streaming.")
        return {"messages": []} # Final output already present in message graph
    except Exception as e:
        logger.error(f"Orchestrator node failed: {str(e)}")
        return {"messages": []}

def build_agent_graph():
    """Builds and compiles the LangGraph workflow."""
    logger.info("Building LangGraph agent workflow.")
    workflow = StateGraph(AgentState)
    
    # Configure the tool execution node
    tool_node = ToolNode([vector_search_tool, web_search_tool])
    
    # Add nodes to graph
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("orchestrator_agent", orchestrator_node)
    
    # Establish edges
    workflow.set_entry_point("research_agent")
    workflow.add_conditional_edges("research_agent", should_continue)
    workflow.add_edge("tools", "research_agent")
    workflow.add_edge("orchestrator_agent", END)
    
    return workflow.compile()
