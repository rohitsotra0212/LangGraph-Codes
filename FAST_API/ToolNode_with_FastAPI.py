import os
import json
import uvicorn
import logging
from typing import Any, List, Literal
from typing_extensions import TypedDict, Annotated

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from fast_payload import app


load_dotenv()
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Set it in your environment or .env file.")


class RAGState(TypedDict):
    query: str
    file_name: str
    route: Literal["RAG", "WEB"]
    file_exists: bool
    chromaDB: Any
    llm: Any
    embeddings: Any
    raw_docs: list
    answer: str
    messages: Annotated[List[Any], add_messages]


class AnswerSchema(BaseModel):
    query: str
    route: str
    answer: str


GLOBAL_RAW_DOCS = []
GLOBAL_CHROMA = None
GLOBAL_LLM = None


def init_resources_node(state: RAGState) -> RAGState:
    logging.info("Init resources node started...")
    global GLOBAL_LLM, GLOBAL_CHROMA

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )

    chroma_db = Chroma(
        persist_directory="./ChromaDB/Syntel",
        collection_name="syntel",
        embedding_function=embeddings,
    )

    GLOBAL_LLM = llm
    GLOBAL_CHROMA = chroma_db

    state["llm"] = llm
    state["embeddings"] = embeddings
    state["chromaDB"] = chroma_db
    state["messages"] = [HumanMessage(content=state["query"])]
    return state


def query_router_node(state: RAGState) -> RAGState:
    query = state["query"].lower()
    keywords = [
        "amount", "policy", "salary", "location", "designation",
        "joining", "period", "office", "offer", "employee", "document"
    ]
    state["route"] = "RAG" if any(k in query for k in keywords) else "WEB"
    logging.info(f"Routing to {state['route']}")
    return state


def check_file_existence_node(state: RAGState) -> RAGState:
    global GLOBAL_RAW_DOCS

    file_name = state.get("file_name", "")
    if not file_name:
        state["file_exists"] = False
        state["raw_docs"] = []
        GLOBAL_RAW_DOCS = []
        logging.info("No file path provided.")
        return state

    existing = state["chromaDB"].get(where={"filename": file_name})
    state["file_exists"] = len(existing.get("ids", [])) > 0

    if state["file_exists"]:
        logging.info("File already indexed. Loading chunks from Chroma.")
        from langchain_core.documents import Document
        state["raw_docs"] = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(existing.get("documents", []), existing.get("metadatas", []))
        ]
    else:
        logging.info("File not indexed. Will ingest.")
        state["raw_docs"] = []

    GLOBAL_RAW_DOCS = state["raw_docs"]
    return state


def ingestion_node(state: RAGState) -> RAGState:
    global GLOBAL_RAW_DOCS, GLOBAL_CHROMA

    file_name = state.get("file_name", "")
    if not file_name or not os.path.exists(file_name):
        logging.info("File not found for ingestion.")
        state["raw_docs"] = []
        GLOBAL_RAW_DOCS = []
        return state

    logging.info("Ingestion node running...")
    loader = PDFPlumberLoader(file_name)
    loaded_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    raw_docs = splitter.split_documents(loaded_docs)

    for i, doc in enumerate(raw_docs):
        doc.metadata.update({
            "department": "HR",
            "year": 2019,
            "filename": file_name,
            "chunk_id": i,
        })

    state["chromaDB"].add_documents(raw_docs)
    state["raw_docs"] = raw_docs
    GLOBAL_RAW_DOCS = raw_docs
    GLOBAL_CHROMA = state["chromaDB"]
    logging.info(f"Indexed {len(raw_docs)} chunks.")
    return state


@tool
def hybrid_retriever(query: str) -> str:
    """Retrieve relevant context from the indexed PDF using Chroma + BM25 hybrid search."""
    global GLOBAL_RAW_DOCS, GLOBAL_CHROMA

    if not GLOBAL_CHROMA or not GLOBAL_RAW_DOCS:
        return ""

    chroma = GLOBAL_CHROMA.as_retriever(search_kwargs={"k": 5})
    bm25 = BM25Retriever.from_documents(GLOBAL_RAW_DOCS)
    bm25.k = 5
    hybrid = EnsembleRetriever(retrievers=[chroma, bm25], weights=[0.7, 0.3])
    retrieved_docs = hybrid.invoke(query)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic mathematical expression."""
    import math
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def web_search(query: str) -> str:
    """Answer general knowledge questions using the language model."""
    global GLOBAL_LLM
    if GLOBAL_LLM is None:
        return "LLM not initialized"
    response = GLOBAL_LLM.invoke([
        SystemMessage(content="You are a concise helpful assistant. Answer clearly in 2-4 lines."),
        HumanMessage(content=query),
    ])
    return response.content


@tool
def save_to_api(final_json: str) -> str:
    """Save final generated JSON output to the FastAPI endpoint."""
    try:
        url = "http://127.0.0.1:8000/save"
        payload = json.loads(final_json)
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
        logging.info(f"Status code: {response.status_code}")
        if response.status_code == 200:
            logging.info("Saved successfully")
            return "Saved Successfully"
        logging.info(f"Save failed: {response.text}")
        return f"Failed: {response.text}"
    except Exception as e:
        return f"API Error: {str(e)}"


TOOLS = [hybrid_retriever, calculator, web_search]
llm_for_agent = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
llm_with_tools = llm_for_agent.bind_tools(TOOLS)
tool_node = ToolNode(TOOLS)
save_tool_node = ToolNode([save_to_api])


def agent_node(state: RAGState) -> RAGState:
    logging.info("Running agent node...")
    system_prompt = SystemMessage(
        content=(
            "You are a Smart AI assistant. "
            "Rules: "
            "1. If the question is about document contents, call hybrid_retriever exactly once. "
            "2. If the question is a math calculation, call calculator. "
            "3. If the question is general knowledge, call web_search. "
            "4. Pass the user's original query exactly as-is to the selected tool. "
            "5. If a tool output is already present, do not call the tool again."
        )
    )
    response = llm_with_tools.invoke([system_prompt] + state["messages"])
    logging.info(f"Agent tool calls: {getattr(response, 'tool_calls', None)}")
    return {"messages": [response]}


def agent_router(state: RAGState) -> str:
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return "generate"


def generate_node(state: RAGState) -> RAGState:
    logging.info("Running generate node...")
    tool_msgs = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
    context = "\n\n".join([msg.content for msg in tool_msgs if getattr(msg, "content", None)])

    if not context.strip():
        result = AnswerSchema(query=state["query"], route=state["route"], answer="Sorry, No Information.")
        return {"answer": json.dumps(result.model_dump(), indent=4)}

    stripped = context.strip()
    if stripped.replace('.', '', 1).isdigit():
        result = AnswerSchema(query=state["query"], route=state["route"], answer=stripped)
        return {"answer": json.dumps(result.model_dump(), indent=4)}

    final_prompt = [
        SystemMessage(
            content=(
                "You are a strict answer generator. Use the provided tool context only. "
                "If the answer is not present, respond with 'Sorry, I don't know.'"
                f"\n\nContext:\n{context}"
            )
        ),
        HumanMessage(content=f"Query: {state['query']}"),
    ]

    structured_llm = state["llm"].with_structured_output(AnswerSchema)
    response = structured_llm.invoke(final_prompt)
    result = AnswerSchema(query=state["query"], route=state["route"], answer=response.answer)
    logging.info("Generated final answer")
    return {"answer": json.dumps(result.model_dump(), indent=4)}



def save_output_node(state: RAGState) -> RAGState:
    logging.info("Saving output to FastAPI...")
    save_result = save_to_api.invoke({"final_json": state["answer"]})
    logging.info(f"Save result: {save_result}")
    
    return {"messages": [AIMessage(content=f"Save result: {save_result}")]}


builder = StateGraph(RAGState)
builder.add_node("init_resources_node", init_resources_node)
builder.add_node("query_router_node", query_router_node)
builder.add_node("check_file_existence_node", check_file_existence_node)
builder.add_node("ingestion_node", ingestion_node)
builder.add_node("agent_node", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("generate_node", generate_node)
builder.add_node("save_output_node", save_output_node)
builder.add_node("save_tools", save_tool_node)

builder.add_edge(START, "init_resources_node")
builder.add_edge("init_resources_node", "query_router_node")

builder.add_conditional_edges(
    "query_router_node",
    lambda state: "RAG" if state["route"] == "RAG" else "WEB",
    {
        "RAG": "check_file_existence_node",
        "WEB": "agent_node",
    },
)

builder.add_conditional_edges(
    "check_file_existence_node",
    lambda state: "retriever" if state["file_exists"] else "ingest",
    {
        "retriever": "agent_node",
        "ingest": "ingestion_node",
    },
)

builder.add_edge("ingestion_node", "agent_node")

builder.add_conditional_edges(
    "agent_node",
    agent_router,
    {
        "tools": "tools",
        "generate": "generate_node",
    },
)

builder.add_edge("tools", "generate_node")
builder.add_edge("generate_node", "save_output_node")
builder.add_edge("save_output_node", "save_tools")
builder.add_edge("save_tools", END)

app = builder.compile()


if __name__ == "__main__":
    # Now run graph
    query = input("Enter your query here: ").strip()

    result = app.invoke(
        {
            "query": query,
            "file_name": r"F:\GEN_AI\Graph_CrewAI\data\Syntel_Offer123.PDF",
            "messages": [HumanMessage(content=query)],
        }
    )

    print(result["answer"])