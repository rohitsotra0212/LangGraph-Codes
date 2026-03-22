Framework used: LangGraph
-------------------------

**Script 1: RAG Search vs Web Search**
-> SharedState(TypedDict)
    query:- user's query
    input_path: RAG knowledge path or external file like PDF
    route: Literal["internal","web"]: it return "internal" if query matches with provided keywords like 'policy', exit policy' otherwise "web"
    file_exists: Check if input file(RAG/PDF file) already ingested or not
    answer: Final answer of the query


