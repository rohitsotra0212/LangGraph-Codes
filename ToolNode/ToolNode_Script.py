import os, json
import logging
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import List, Literal, Any
from typing_extensions import TypedDict, Annotated

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ---------------- STATE ----------------
class RAGState(TypedDict):
    query: str
    file_name: str
    route: Literal["rag","web"]
    file_exists: bool
    chromaDB: Any
    llm: Any
    embeddings: Any
    raw_docs: str
    answer: str
    messages: Annotated[List[Any], add_messages]

class AnswerSchema(BaseModel):
    query: str
    route: str
    answer: str

# ---------------- GLOBALS (for tool) ----------------
GLOBAL_RAW_DOCS = None
GLOBAL_CHROMA = None
GLOBAL_LLM = None

# ---------------- NODE 1 ----------------
def init_resources_node(state: RAGState) -> RAGState:
    logging.info("Init Resources node started...")
    global GLOBAL_LLM, GLOBAL_CHROMA, GLOBAL_RAW_DOCS

    state["messages"] = [HumanMessage(content=state["query"])]

    state["llm"] = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    state["embeddings"] = OpenAIEmbeddings(model="text-embedding-3-small")

    state["chromaDB"] = Chroma(
        persist_directory="./ChromaDB/Syntel",
        collection_name="syntel",
        embedding_function=state["embeddings"]
    )
            
    GLOBAL_CHROMA = state["chromaDB"]
    GLOBAL_LLM = state["llm"]

    return state

# ---------------- NODE 2 ----------------
def query_router_node(state: RAGState) -> RAGState:
    
    query = state["query"].lower()
    keywords = ["amount","policy","salary","location","designation","joining","period","office"]

    state["route"] = "rag" if any(k in query for k in keywords) else "web"
    logging.info(f"Routing to {state['route']}")
    return state

# ---------------- NODE 3 ----------------
def check_file_existance_node(state: RAGState) -> RAGState:

    global GLOBAL_LLM, GLOBAL_CHROMA, GLOBAL_RAW_DOCS
    file_count = state["chromaDB"].get(where={"filename": state["file_name"]})
    state["file_exists"] = len(file_count["ids"]) > 0

    if state["file_exists"] or not state["file_name"]:
        logging.info(f"File Exists --> Retreiver")
        raw_docs = state["chromaDB"].get(where={"filename": state["file_name"]})
         
        from langchain_core.documents import Document
        state["raw_docs"] = [Document(page_content=text, metadata=meta) for text, meta in zip(raw_docs["documents"], raw_docs["metadatas"])]

    else:
        state["file_exists"] = False
        logging.info(f"File Not Exists/Not Found --> Ingestion")

    GLOBAL_RAW_DOCS = state["raw_docs"]
    
    return state


# ---------------- NODE 4 ----------------
def ingestion_node(state: RAGState) -> RAGState:

    global GLOBAL_LLM, GLOBAL_CHROMA, GLOBAL_RAW_DOCS

    logging.info("Ingestion Node Running..")

    if state["file_name"]:
        loader = PDFPlumberLoader(state["file_name"])
        loaded_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        raw_docs = splitter.split_documents(loaded_docs)

        for i, doc in enumerate(raw_docs):
            doc.metadata.update({
                "department": "HR",
                "year": 2019,
                "filename": state["file_name"],
                "chunk_id": i
            })

        state["chromaDB"].add_documents(raw_docs)
        state["raw_docs"] = raw_docs

        GLOBAL_CHROMA = state["chromaDB"]
        GLOBAL_RAW_DOCS = state["raw_docs"]

    else:
        logging.info("File Not Found")

    return state

@tool
def hybrid_retriever(query: str) -> str:
    """Retrieve relevant context from the Chroma vector store for the exact user query"""
    global GLOBAL_LLM, GLOBAL_CHROMA, GLOBAL_RAW_DOCS

    logging.info(f"Tool called: hybrid_retriever('{query}'")
    
    chroma = GLOBAL_CHROMA.as_retriever(search_kwargs={"k": 5})
    bm25 = BM25Retriever.from_documents(GLOBAL_RAW_DOCS)
    bm25.k = 5

    hybrid = EnsembleRetriever(retrievers=[chroma, bm25], weights=[0.7,0.3])

    retrieved_docs = hybrid.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    logging.info(f"Context length: {len(context)}")

    return context

@tool
def calculator(expression: str) -> str:
    """Evaluate basic mathematical expressions like 2+2, 10*5, 100/4"""
    import math

    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    
@tool
def web_search(query: str) -> str:
    """Generate response using llm"""
    global GLOBAL_LLM
    response = GLOBAL_LLM.invoke(query)

    return response.content
    
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

tools = [hybrid_retriever,calculator,web_search]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

def agent_node(state: RAGState) -> RAGState:
    logging.info("Running Agent Node..")

    system_prompt = SystemMessage(content= """
                                You are a Smart AI assistant.
                           
                           RULES:
                            - If the question is about documents → call hybrid_retriever exactly once.
                            - If the question is a math calculation → call calculator.
                            - If the question is abount general purpose or about news related → call web_search.
                            - Pass the user's original query exactly as-is to the tool.
                            - Use only the retrieved context to answer factual RAG questions.
                            - If the answer is not present in the retrieved context, say exactly: I Don't Know
                            - If tool output is already available, do not call the tool again.
                           """)
    
    response = llm_with_tools.invoke([system_prompt] + state["messages"])
    logging.info(f"Agent response type: {type(response)}")
    logging.info(f"Agent tool calls: {getattr(response, 'tool_calls', None)}")

    return {"messages": [response]}

def agent_router(state: RAGState) -> RAGState:
    last_msg = state["messages"][-1]

    if getattr(last_msg, "tool_calls",None):
        logging.info("Calling Tools..")
        return "tools"
    
    logging.info("Calling Generate Node..")
    return "generate"

def generate_node(state: RAGState) -> RAGState:
    logging.info("Running Generate Node...")

    tool_msg = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]

    context = "\n\n".join([doc.content for doc in tool_msg if doc.content])

    logging.info(f"Tool messages found: {len(tool_msg)}")
    logging.info(f"Context length in generate: {len(context)}")

    if not context:
        logging.info("No context found. Returning I Don't Know")
        return {"answer": "Sorry, I Don't Know"}
    elif context.strip().replace('.', '', 1).isdigit():
        logging.info("Detected calculator result → returning directly")

        result = AnswerSchema(
            query=state["query"],
            route=state["route"],
            answer=context.strip())
        
        return {"answer": json.dumps(result.model_dump(), indent=4)}
        
    else:
        final_prompt = [
            SystemMessage(content= f"""
                        You are a strict RAG assistant.
                        Use provided context only to generate answer.
                        If answer is not present in the provided context then say, 'Sorry, I don't know.'
                          
                        Context:
                        {context}
                        """),
            HumanMessage(content=f"""Query: {state['query']}""")
        ]

        structured_llm = state["llm"].with_structured_output(AnswerSchema)
        response = structured_llm.invoke(final_prompt)
        logging.info("Generated final answer")

    result = AnswerSchema(
        query=state["query"],
        route=state["route"],
        answer=response.answer 
    )

    return {"answer": json.dumps(result.model_dump(), indent=4)}

"""
def web_search_node(state: RAGState) -> RAGState:
    logging.info("Running web_search/general LLM node")

    structured_llm = llm_with_tools.with_structured_output(AnswerSchema)
    response = structured_llm.invoke([HumanMessage(content=state["query"])])

    result = AnswerSchema(
        query=state["query"],
        route=state["route"],
        answer=response.answer 
    )

    return {"answer": json.dumps(result.model_dump(), indent=4)}
"""

# --------------------------------------------------
# Build graph
# --------------------------------------------------
builder = StateGraph(RAGState)

builder.add_node("init_resources_node", init_resources_node)
builder.add_node("query_router_node", query_router_node)
builder.add_node("check_file_existance_node",check_file_existance_node)
builder.add_node("ingestion_node",ingestion_node)
builder.add_node("tools", tool_node)
builder.add_node("agent_node", agent_node)
builder.add_node("generate_node", generate_node)
#builder.add_node("web_search_node", web_search_node)

builder.add_edge(START, "init_resources_node")
builder.add_edge("init_resources_node", "query_router_node")
builder.add_conditional_edges("query_router_node", lambda state: "rag" if state["route"] == "rag" else "web",
                              {
                                  "rag": "check_file_existance_node",
                                  "web": "agent_node"
                              })
builder.add_conditional_edges("check_file_existance_node", lambda state: "retriever" if state["file_exists"] else "ingestion",
                              {
                                  "retriever": "agent_node",
                                  "ingestion": "ingestion_node"
                              })

builder.add_conditional_edges("agent_node", agent_router,
                              {
                                  "tools": "tools",
                                  "generate": "generate_node"
                              })
builder.add_edge("ingestion_node", "agent_node")
builder.add_edge("tools", "generate_node")
builder.add_edge("generate_node", END)
#builder.add_edge("web_search_node", END)

app = builder.compile()

#print(app.get_graph().draw_ascii())   # ASCII diagram in console
#app.get_graph().draw_png("workflow.png")  # Requires graphviz installed


if __name__ == "__main__":
    query = input("Enter your query here: ")

    result = app.invoke({
        "query": query,
        "file_name": r"F:\GEN_AI\Graph_CrewAI\data\Syntel_Offer123.PDF",
        "messages": [HumanMessage(content= query)]
    })

    print("\nFinal Answer: ")
    print(result["answer"])