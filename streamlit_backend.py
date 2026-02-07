import os, tempfile, uuid, operator, json
from typing import TypedDict, Annotated, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# globals


# Session state
current_thread_id: Optional[str] = None

# Vector stores per session
vector_stores = {}

# Qdrant client
client = QdrantClient(path="./qdrant_data")

# Models
embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# session handling
def init_session():
    """Initialize or retrieve current session thread ID."""
    global current_thread_id

    if current_thread_id is None:
        current_thread_id = str(uuid.uuid4())

    return current_thread_id


def get_pdf_collection():
    """Get collection name for PDF documents in current session."""
    return f"pdf_{current_thread_id}"


def get_yt_collection()
    """Get collection name for YouTube transcripts in current session."""
    return f"yt_{current_thread_id}"


# collection factory
def get_vector_store(collection_name):
    """
    Get or create a Qdrant vector store for the given collection.
    
    Args:
        collection_name: Name of the collection to retrieve or create
        
    Returns:
        QdrantVectorStore instance
    """
     
    if not client.collection_exists(collection_name):

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )

    return QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=collection_name
    )

# pdf ingestion
def ingest_pdf(uploaded_file):
    """
    Get or create a Qdrant vector store for the given collection.
    
    Args:
        collection_name: Name of the collection to retrieve or create
        
    Returns:
        QdrantVectorStore instance
    """
    init_session()

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splits = splitter.split_documents(docs)

        splits = [d for d in splits if d.page_content.strip()]

        if not splits:
            raise ValueError("No text in PDF")

        collection = get_pdf_collection()

        store = get_vector_store(collection)

        store.add_documents(splits)

        vector_stores[collection] = store

        return len(splits)

    finally:
        if tmp_path:
            os.unlink(tmp_path)


# youtube ingestion
def ingest_youtube(video_id):
    """
    Fetch and ingest YouTube video transcript into the vector store.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        int: Number of document chunks created
        
    Raises:
        Exception: If transcript cannot be fetched
    """
    init_session()

    print(f"in the yt backend {video_id}")

    transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    # print(f"in the yt backend transcript {transcript}")


    # text = " ".join(t["text"] for t in transcript)
    text_parts = []
    for t in transcript:
        if isinstance(t, dict):
            text_content = t.get("text", "")
        else:
            text_content = getattr(t, 'text', "")
        
        if text_content:  # Skip empty strings
            text_parts.append(str(text_content))
    
    text = " ".join(text_parts)

    docs = splitter.create_documents([text])

    collection = get_yt_collection()
    print(f"in the yt backend docs {text} and collection {collection}")

    store = get_vector_store(collection)

    store.add_documents(docs)

    vector_stores[collection] = store

    # print(f"in the yt len docs {len(docs)}")

    return len(docs)


# unified retriever
def search_collection(collection, query, k=4):
    """
    Search a collection for relevant documents.
    
    Args:
        collection: Name of the collection to search
        query: Search query string
        k: Number of results to return (default: 4)
        
    Returns:
        List of relevant documents or None if collection doesn't exist
    """

    if collection not in vector_stores:

        if not client.collection_exists(collection):
            return None

        vector_stores[collection] = get_vector_store(collection)

    store = vector_stores[collection]

    return store.similarity_search(query, k=k)



# toolsl (pdf + yt)
@tool
def search_pdf(query: str):
    """
    Search uploaded PDF documents for relevant information.
    
    Use this tool when the user asks questions about uploaded PDF files.
    Returns contextual information from the most relevant PDF sections.
    
    Args:
        query: The search query or question about PDF content
        
    Returns:
        dict: Contains 'content' key with relevant PDF text or error message
    """

    init_session()

    collection = get_pdf_collection()

    docs = search_collection(collection, query)

    if not docs:
        return {"content": "No PDF data found."}

    context = "\n\n".join(d.page_content for d in docs)

    return {"content": context}


@tool
def search_youtube(query: str):
    """
    Search YouTube video transcripts for relevant information.
    
    Use this tool when the user asks questions about YouTube videos they've added.
    Returns contextual information from the most relevant transcript sections.
    
    Args:
        query: The search query or question about video content
        
    Returns:
        dict: Contains 'content' key with relevant transcript text or error message
    """
    
    init_session()

    collection = get_yt_collection()

    docs = search_collection(collection, query)

    if not docs:
        return {"content": "No YouTube data found."}

    context = "\n\n".join(d.page_content for d in docs)

    return {"content": context}


tools = [search_pdf, search_youtube]

llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list, operator.add]


def chat_node(state: ChatState):

    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


tool_node = ToolNode(tools)


def route(state: ChatState):

    last = state["messages"][-1]

    if getattr(last, "tool_calls", None):
        return "tools"

    return "end"


graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", route, {
    "tools": "tools",
    "end": END
})

graph.add_edge("tools", "chat")

chatbot = graph.compile(
    checkpointer=InMemorySaver()
)



