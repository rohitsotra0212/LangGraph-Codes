import os
import json

from hashlib import md5
from typing import TypedDict, List, Literal
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, END

class SharedState(TypedDict):
    query: str
    input_path: str
    route: Literal["internal","web"]
    file_exists: bool
    answer: str

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key= os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings()
chromaDb = Chroma(persist_directory="./ChromaDB",
                  collection_name="wipro",
                  embedding_function= embeddings)

## Node 1:
def query_validator_node(state: SharedState) -> SharedState:
    print("Runing query_validator_node..")

    query = state["query"].lower()
    keywords = ["policy","amount","wipro","exit","salary"]

    if any(word in query for word in keywords):
        state["route"] = "internal"
    else:
        state["route"] = "web"
    
    print(f"query_validator_node output: {state}")

    return state

## Node 2:
def check_vector_store_node(state: SharedState) -> SharedState:
    print("Running check_vector_store_node...")

    #if not state["input_path"]:
    #    state["file_exists"] = False
    #    return state
    
    results = chromaDb.get(where={"filename": state["input_path"]})
    state["file_exists"] = len(results["ids"]) > 0

    print(f"check_vector_store_node output:{state['file_exists']}")
    return state

## Node 3:
def ingest_knowledgeBase_node(state: SharedState) -> SharedState:
    print("Running ingest_knowledgeBase_node..")

    loader = PDFPlumberLoader(state["input_path"])
    loaded = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    docs = splitter.split_documents(loaded)

    file_id = md5(state["input_path"].encode()).hexdigest()
    
    # Update each doc’s metadata
    for i, doc in enumerate(docs):
        doc.metadata.update({"filename": state["input_path"], "chunk": i})

    ids = [f"{file_id}_{i}" for i in range(len(docs))]
    #print(f" --------> {chromaDb.get()['metadatas'][:2]}")
    chromaDb.add_documents(documents= docs,
                           ids=ids
                           )

    state["file_exists"] = True

    print(f"ingest_knowledgeBase_node output: {state}")

    return state

## Node 4:
def internal_search_node(state: SharedState) -> SharedState:
    print("Running internal_search_node...")

    retriever = chromaDb.as_retriever(search_kwargs={"k":3})

    #retrieved_docs1 =retriever._get_relevant_documents(state["query"])
    retrieved_docs = retriever.invoke(state["query"])

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = PromptTemplate(input_variables=["query","context"],
                            template= """
                                    You are a helpful AI Assistant.
                                    Use only provided context to generate the answer.
                                    If answer is not present in the provided context then say, 'I don't know'.

                                    Query:
                                    {query}

                                    Context:
                                    {context}

                                    Answer:
                                """)
    final_prompt = prompt.format( query = state["query"], context = context)

    response = llm.invoke(final_prompt)
    state["answer"] = response.content

    print(f"internal_search_node ouput: {state}")

    return state

## Node 5:
def web_search_node(state: SharedState) -> SharedState:

    response = llm.invoke(state["query"])
    state["answer"] = response.content
    return state

## Node 6:
def answer_node(state: SharedState) -> SharedState:
    return state

builder = StateGraph(SharedState)

builder.add_node("Query_Validator",query_validator_node)
builder.add_node("Check_Vector_Store",check_vector_store_node)
builder.add_node("Ingest_KnowledgeBase",ingest_knowledgeBase_node)
builder.add_node("Internal_Search",internal_search_node)
builder.add_node("Web_Search",web_search_node)
builder.add_node("Answer",answer_node)

builder.set_entry_point("Query_Validator")

builder.add_conditional_edges("Query_Validator",
                              lambda state: state["route"],
                              {
                                  "web":"Web_Search",
                                  "internal": "Check_Vector_Store"
                              })


builder.add_conditional_edges("Check_Vector_Store",
                              lambda state: "Internal_Search" if state["file_exists"] else "Ingest_KnowledgeBase" ,
                              {
                                  "Ingest_KnowledgeBase":"Ingest_KnowledgeBase",
                                  "Internal_Search": "Internal_Search"
                              })

builder.add_edge("Ingest_KnowledgeBase","Internal_Search")
builder.add_edge("Internal_Search","Answer")
builder.add_edge("Web_Search","Answer")
builder.add_edge("Answer",END)

app = builder.compile()

#print(app.get_graph().draw_ascii())   # ASCII diagram in console
#app.get_graph().draw_png("workflow.png")  # Requires graphviz installed

if __name__ == "__main__":
    result = app.invoke({
        "query": input("Enter your Query: "),
        "input_path": r"F:\GEN_AI\Graph_CrewAI\data\Offer_Letter.pdf"
    })

    print(f"AI Response:\n{result['answer']}")











