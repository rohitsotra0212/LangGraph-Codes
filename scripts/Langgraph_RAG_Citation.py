import os

from hashlib import md5
from typing import TypedDict
from typing_extensions import Literal
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

## Retrievers Dense & Sparse
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever #EnsembleRetriever  

from langgraph.graph import StateGraph, START, END

class StateSchema(TypedDict):
    query: str
    input_file: str
    route: Literal["internal","web"]
    file_exists: bool
    answer: str
    raw_docs: str
    retriever_type: Literal["Chroma","BM25","Hybrid"]
    retriever_selected: str
    llm: str
    context: str
    sources: str

## Node 1
def query_validator_node(state: StateSchema):
  query = state["query"].lower()
  keywords = ["wipro","salary","amount","exit","key","pair","aws", "ec2"]

  if any(word in query for word in keywords):
    state["route"] = "internal"
  else:
    state["route"] = "web"

  print(f"Routing to : {state['route']}")
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key = os.getenv("OPENAI_API_KEY"))

  state["llm"] = llm

  return state

## Node 2:
def check_file_exists_in_vectorStore_node(state: StateSchema):

  print("Running check_file_exists_in_vectorStore_node") 
  embeddings = OpenAIEmbeddings()
  chromaDB = Chroma(persist_directory="./ChromaDB",collection_name="internal_wipro", embedding_function= embeddings)

  results = chromaDB.get(where={"filename": state["input_file"]})
  state["file_exists"] = len(results["ids"]) > 0

  print(f"File Exists? : {state['file_exists']}")
  return state

## Node 3:
def data_ingestion_node(state: StateSchema):
  print("Data ingestion in progress...")
  loader = PDFPlumberLoader(state["input_file"])
  loaded = loader.load()

  spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  state["raw_docs"] = spliter.split_documents(loaded)
  
  file_id = md5(state["input_file"].encode()).hexdigest()

  embeddings = OpenAIEmbeddings()
  chromaDB = Chroma(persist_directory="./ChromaDB",collection_name="internal_wipro", embedding_function= embeddings)  

  # Update each doc’s metadata
  for i, doc in enumerate(state["raw_docs"]):
    doc.metadata.update({"filename": state["input_file"], "chunk": i})

  ids = [f"{file_id}_{i}" for i in range(len(state["raw_docs"]))]

  chromaDB.add_documents(documents= state["raw_docs"],ids=ids)

  print("File has been ingested successfully!!")
  state["file_exists"] = True

  return state

## Node:
def select_retriever_node(state: StateSchema):

  embeddings = OpenAIEmbeddings()
  chromaDB = Chroma(persist_directory="./ChromaDB",collection_name="internal_wipro", embedding_function= embeddings)
  #chromaDB.add_documents(state["raw_docs"])
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key = os.getenv("OPENAI_API_KEY"))

  state["llm"] = llm

  if state["file_exists"] and state["retriever_type"] == "Chroma":
    state["retriever_selected"] = chromaDB.as_retriever(search_kwargs={"k":3})
  elif state["file_exists"] and state["retriever_type"] == "BM25":
    state["retriever_selected"] = BM25Retriever.from_documents(state["raw_docs"])
    state["retriever_selected"].k = 3  
  else:
    state["retriever_selected"] = chromaDB.as_retriever(search_kwargs={"k":3})

  return state

## Node 4:
def rag_search_node(state: StateSchema):
  print("Running RAG Search Node...")

  state["answer"] = "RAG Search Completed"
  return state

## Node 5:
def dense_retriever_node(state: StateSchema):
  print("[INFO:] Chroma Retriever Selected..")
  
  if state["retriever_type"] == "Chroma":
    retrieved_docs = state["retriever_selected"].invoke(state["query"])

    formatted_docs = []
    sources = []
    for i, doc in enumerate(retrieved_docs):
      source_id = f"[DOC_{i+1}]"
      content = doc.page_content

      formatted_docs.append(f"{source_id} {content}")
      sources.append({
                "id": source_id,
                "content": content,
                "metadata": doc.metadata
            })

    context = "\n\n".join(formatted_docs)  
    #context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Strong citation prompt
    prompt = PromptTemplate(input_variables=[ "query" , "context"],
                            template= """
                                      You are a strict RAG assistant.
                                      RULES:
                                        - Answer ONLY using the provided context
                                        - DO NOT use prior knowledge
                                        - If answer is not in context → say "I don't know"
                                        - You MUST cite sources using [document_X]
                                        - Do NOT answer without citation

                                      Query:
                                      {query}

                                      Context:
                                      {context}

                                      Output Format:
                                      Answer: <your answer>
                                      Sources: <list of DOC IDs used>
                                      Context: <context>
                            """)
    
    final_prompt = prompt.format(query= state["query"], context= context)
    response = state["llm"].invoke(final_prompt)
    state["answer"] = response.content
    state["context"] = context
    state["sources"] = sources

  return state

## Node 6:
def sparse_retriever_node(state: StateSchema):
  print("[INFO:] BM25 Retriever Selected.")

  if state["retriever_type"] == "BM25":
    retrieved_docs = state["retriever_selected"].invoke(state["query"])
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = PromptTemplate(input_variables=[ "query" , "context"],
                            template= """
                                      You are a strict RAG assistant.
                                      Use only provided context to generate answer.
                                      If answer is not present in provided context that say. 'I Don't Know'

                                      Query:
                                      {query}

                                      Context:
                                      {context}

                                      Answer:
                            """)
    
    final_prompt = prompt.format(query= state["query"], context= context)
    response = state["llm"].invoke(final_prompt)
    state["answer"] = response.content
    

## Node 7:
def hybrid_retriever_node(state: StateSchema):
  print("Running Hybrid Retriver Node...")

  if state["retriever_type"] == "Hybrid":
    state["answer"] = "Hybrid Retriever Completed"
  return state

## Node 8:
def llm_search_node(state: StateSchema):
  print("Running LLM Search Node...")

  response = state["llm"].invoke(state["query"])
  state["answer"] = response.content

  return state

## Node 9:
def answer_node(state: StateSchema):
  print("Final Answer Node...")
  return state

builder = StateGraph(StateSchema)

builder.add_node("query_validator",query_validator_node)
builder.add_node("check_file_exists_in_vectorStore",check_file_exists_in_vectorStore_node)
builder.add_node("rag_search",rag_search_node)
builder.add_node("select_retriever",select_retriever_node)
builder.add_node("data_ingestion",data_ingestion_node)
builder.add_node("Chroma",dense_retriever_node)
builder.add_node("BM25",sparse_retriever_node)
builder.add_node("Hybrid",hybrid_retriever_node)
builder.add_node("llm_search",llm_search_node)
builder.add_node("answer",answer_node)

builder.set_entry_point("query_validator")
builder.add_conditional_edges("query_validator", lambda state: state["route"],
                              {
                                  "internal": "rag_search",
                                  "web": "llm_search"
                              })

builder.add_edge("rag_search","check_file_exists_in_vectorStore")

builder.add_conditional_edges("check_file_exists_in_vectorStore", lambda state: state["file_exists"],
                              {
                                  True: "select_retriever",
                                  False: "data_ingestion"

                              })
builder.add_edge("data_ingestion","select_retriever")

builder.add_conditional_edges("select_retriever", 
                              lambda state: 
                              "dense" if state["retriever_type"]  == "Chroma" else
                              "sparse" if state["retriever_type"] == "BM25" else
                              "hybrid" if state["retriever_type"] == "Hybrid"
                              else "dense", ## Default Retriever
                              {
                                  "dense": "Chroma",
                                  "sparse": "BM25",
                                  "hybrid": "Hybrid"
                              })

builder.add_edge("Chroma","answer")
builder.add_edge("BM25","answer")
builder.add_edge("Hybrid","answer")
builder.add_edge("llm_search","answer")
builder.add_edge("answer", END)
app = builder.compile()

if __name__ == "__main__":
  
  result = app.invoke(input=
                      {
                      "query": input("Enter your query here: "),
                      "input_file": r"F:\GEN_AI\Graph_CrewAI\data\Hadoop1.pdf",
                      "retriever_type": input("Enter retriever type (Chroma / BM25 / Hybrid): ")
                      })
  
  print(result["answer"])
