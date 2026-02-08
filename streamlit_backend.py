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
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from urllib.parse import urlparse, parse_qs


import requests

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


def get_yt_collection():
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




def extract_video_id(url: str) -> str:

    parsed = urlparse(url)

    if "youtube.com" in parsed.netloc:
        vid = parse_qs(parsed.query).get("v")
        if vid:
            return vid[0]

    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    


# # pdf ingestion
# def ingest_pdf(uploaded_file):
#     """
#     Get or create a Qdrant vector store for the given collection.
    
#     Args:
#         collection_name: Name of the collection to retrieve or create
        
#     Returns:
#         QdrantVectorStore instance
#     """
#     init_session()

#     tmp_path = None

#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
#             f.write(uploaded_file.read())
#             tmp_path = f.name

#         loader = PyPDFLoader(tmp_path)
#         docs = loader.load()

#         splits = splitter.split_documents(docs)

#         splits = [d for d in splits if d.page_content.strip()]

#         if not splits:
#             raise ValueError("No text in PDF")

#         collection = get_pdf_collection()

#         store = get_vector_store(collection)

#         store.add_documents(splits)

#         vector_stores[collection] = store

#         retrieved = store.similarity_search("what is the topic of pdf", k=5)

#         context = "\n\n".join(d.page_content for d in retrieved)

#                 # 🤖 Ask LLM
#         prompt = f"""
#         Answer the question using only the context below.

#         Context:
#         {context}

#         Question:
#         {"topic of the pdf"}
#         """

#         response = llm.invoke(prompt)
#         print(response.content)

#         url = "https://google.serper.dev/videos"

#         payload = {"q": response.content}

#         headers = {
#             "X-API-KEY": "1b3dcdb7ee679d34ce513e2a1177db3f81144a32",
#             "Content-Type": "application/json"
#         }

#         res = requests.post(url, json=payload, headers=headers)
#         data = res.json()

#         videos = data.get("videos", [])

#         formatted = []

#         for v in videos[:5]:  # limit results
#             formatted.append({
#                 "title": v.get("title"),
#                 "link": v.get("link"),
#                 "channel": v.get("channel"),
#                 "duration": v.get("duration"),
#             })

#         for video in videos:

#             url = video.get("link")

#             if not url:
#                 continue

#             video_id = extract_video_id(url)

#             if not video_id:
#                 continue

#             try:
#                 transcript = YouTubeTranscriptApi().fetch(
#                     video_id,
#                     languages=["en"]
#                 )

#                 text_parts = []

#                 for item in transcript:
#                     text = item.get("text", "").strip()

#                     if text:
#                         text_parts.append(text)

#                 full_text = " ".join(text_parts)

#                 if not full_text:
#                     raise ValueError("Empty transcript")

#                 docs = splitter.create_documents([full_text])

#                 collection = get_yt_collection()

#                 store = get_vector_store(collection)
#                 store.add_documents(docs)

#                 vector_stores[collection] = store

#                 success.append(video_id)

#                 print(f"✅ Ingested {video_id}")
    
#         return len(splits)

#     finally:
#         if tmp_path:
#             os.unlink(tmp_path)


import os
import tempfile
import requests
from youtube_transcript_api import YouTubeTranscriptApi

# ... existing imports for langchain, qdrant, etc ...

def ingest_pdf(uploaded_file):
    """
    Ingest PDF and related YouTube videos into Qdrant vector store.
    """
    init_session()
    tmp_path = None

    try:
        # 1. Ingest PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splits = splitter.split_documents(docs)
        splits = [d for d in splits if d.page_content.strip()]

        if not splits:
            raise ValueError("No text in PDF")

        # Store PDF embeddings
        pdf_collection_name = get_pdf_collection()
        pdf_store = get_vector_store(pdf_collection_name)
        pdf_store.add_documents(splits)
        vector_stores[pdf_collection_name] = pdf_store

        # 2. Search for Context
        retrieved = pdf_store.similarity_search("what is the topic of pdf", k=5)
        context = "\n\n".join(d.page_content for d in retrieved)

        # 🤖 Ask LLM for Topic
        prompt = f"""
        Answer the question using only the context below.
        Context: {context}
        Question: "topic of the pdf"
        """
        response = llm.invoke(prompt)
        topic = response.content
        print(f"Topic identified: {topic}")

        # 3. Fetch Related Videos
        url = "https://google.serper.dev/videos"
        payload = {"q": topic}
        headers = {
            "X-API-KEY": "1b3dcdb7ee679d34ce513e2a1177db3f81144a32",
            "Content-Type": "application/json"
        }

        res = requests.post(url, json=payload, headers=headers)
        videos = res.json().get("videos", [])

        cnt = 0
        # 4. Ingest Videos
        for video in videos:
            print(f"video topis : {video.get('title')}")
            video_url = video.get("link")
            if not video_url:
                continue

            video_id = extract_video_id(video_url)
            if not video_id:
                continue

            try:
                transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
                print("dsklf;asdfj;a : {transcript}")                
                # Combine transcript into one text block or keeping parts based on preference
                text_parts = []
                for t in transcript:
                    if isinstance(t, dict):
                        text_content = t.get("text", "")
                    else:
                        text_content = getattr(t, 'text', "")
                    
                    if text_content:  # Skip empty strings
                        text_parts.append(str(text_content))
                
                text = " ".join(text_parts)


                # Create Documents
                video_docs = splitter.create_documents([text])

                # --- CHANGE START ---
                # Use video_id to create a unique collection key
                # This effectively "appends" this video's store to your dictionary
                
                # Option A: Unique collection per video (Separated Data)
                yt_collection_name = f"yt_{video_id}"
                
                # Option B: Shared collection (Aggregated Data), but keyed by ID in dict
                # yt_collection_name = get_yt_collection() 

                yt_store = get_vector_store(yt_collection_name)
                yt_store.add_documents(video_docs)

                # Store in dictionary using video_id instead of appending to 'success' list
                vector_stores[yt_collection_name] = yt_store
                
                print(f"✅ Ingested video: {video_id} into collection: {yt_collection_name}")
                cnt=cnt+1
                # --- CHANGE END 
                if(cnt > 2):
                    break

            except Exception as e:
                print(f"Skipping video {video_id}: {e}")
                continue
    
        return len(splits)

    finally:
        if tmp_path and os.path.exists(tmp_path):
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
def recommend_youtube(topic: str) -> dict:
    """
    Recommend YouTube videos for a given topic.
    """

    url = "https://google.serper.dev/videos"

    payload = {"q": topic}

    headers = {
        "X-API-KEY": "1b3dcdb7ee679d34ce513e2a1177db3f81144a32",
        "Content-Type": "application/json"
    }

    res = requests.post(url, json=payload, headers=headers)
    data = res.json()

    videos = data.get("videos", [])

    formatted = []

    for v in videos[:5]:  # limit results
        formatted.append({
            "title": v.get("title"),
            "link": v.get("link"),
            "channel": v.get("channel"),
            "duration": v.get("duration"),
        })

    return {
        "topic": topic,
        "results": formatted
    }


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


search = DuckDuckGoSearchRun()

tools = [search_pdf, search_youtube, search, recommend_youtube]

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


