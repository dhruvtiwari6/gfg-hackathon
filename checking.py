import collections
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import os
import tempfile
import operator
from typing import TypedDict, Annotated

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
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain_community.utilities import GoogleSerperAPIWrapper
from urllib.parse import urlparse, parse_qs


import requests

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="RAG Chatbot API",
    description="Multi-session RAG chatbot supporting multiple PDFs and YouTube videos per session",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SessionResponse(BaseModel):
    thread_id: str
    message: str
    status: str = "success"

class DocumentInfo(BaseModel):
    """Info about an ingested document"""
    doc_id: str
    filename: str
    chunks: int
    collection_name: str

class PDFIngestResponse(BaseModel):
    thread_id: str
    doc_id: str
    filename: str
    chunks_created: int
    total_pdfs: int
    message: str
    status: str = "success"

class YouTubeIngestRequest(BaseModel):
    video_id: str = Field(..., description="YouTube video ID")
    title: Optional[str] = Field(None, description="Optional video title for reference")

class YouTubeIngestResponse(BaseModel):
    thread_id: str
    doc_id: str
    video_id: str
    chunks_created: int
    total_videos: int
    message: str
    status: str = "success"

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    thread_id: str = Field(..., description="Session thread ID")

class ChatResponse(BaseModel):
    thread_id: str
    response: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    status: str = "success"

class SessionStatusResponse(BaseModel):
    thread_id: str
    total_pdfs: int
    total_youtube: int
    total_pdf_chunks: int
    total_youtube_chunks: int
    pdf_documents: List[DocumentInfo]
    youtube_documents: List[DocumentInfo]
    message: str


sessions: Dict[str, Dict[str, Any]] = {}
vector_stores: Dict[str, QdrantVectorStore] = {}

client = QdrantClient(path="./qdrant_data")
embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def create_session(thread_id: str = None) -> str:
    """Create a new session"""
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    if thread_id not in sessions:
        sessions[thread_id] = {
            "created_at": None,
            "pdf_docs": [],
            "youtube_docs": []
        }
    
    return thread_id

def get_collection_name(thread_id: str, doc_type: str, doc_id: str) -> str:
    """Get collection name for a specific document"""
    return f"{doc_type}_{thread_id}_{doc_id}"

def get_vector_store(collection_name: str) -> QdrantVectorStore:
    """Get or create a Qdrant vector store"""
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

def search_all_collections(collections: List[str], query: str, k: int = 4):
    """Search across multiple collections and aggregate results"""
    all_results = []
    
    for collection in collections:
        if collection not in vector_stores:
            if not client.collection_exists(collection):
                continue
            vector_stores[collection] = get_vector_store(collection)
        
        store = vector_stores[collection]
        results = store.similarity_search(query, k=k)
        all_results.extend(results)
    
    # Sort by relevance if we have results
    # For now, just return all (you could add scoring here)
    return all_results[:k * 2]  # Return up to k*2 results total


# def ingest_pdf_file(file_content: bytes, filename: str, thread_id: str) -> Dict[str, Any]:
#     """Ingest a single PDF file"""
#     doc_id = str(uuid.uuid4())[:8]  # Short ID
#     tmp_path = None

#     print("Thread ID:", thread_id)
#     print("File:", filename)
#     # print(file_content)

#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
#             f.write(file_content)
#             tmp_path = f.name
        
#         loader = PyPDFLoader(tmp_path)
#         docs = loader.load()
#         splits = splitter.split_documents(docs)
#         splits = [d for d in splits if d.page_content.strip()]
        
#         if not splits:
#             raise ValueError("No text content found in PDF")
        
#         collection = get_collection_name(thread_id, "pdf", doc_id)
#         store = get_vector_store(collection)
#         store.add_documents(splits)
#         vector_stores[collection] = store
        
#         # Store document info
#         doc_info = {
#             "doc_id": doc_id,
#             "filename": filename,
#             "chunks": len(splits),
#             "collection": collection
#         }

#         # if thread_id not in sessions:
#         sessions[thread_id] = {"pdf_docs": []}
#         sessions[thread_id]["pdf_docs"].append(doc_info)
        
#         return {
#             "doc_id": doc_id,
#             "chunks": len(splits),
#             "total_pdfs": len(sessions[thread_id]["pdf_docs"])
#         }
#     except Exception as e:
#         raise ValueError(f"Failed to ingest PDF file: {str(e)}")
#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.unlink(tmp_path)


def extract_video_id(url: str) -> str:

    parsed = urlparse(url)

    if "youtube.com" in parsed.netloc:
        vid = parse_qs(parsed.query).get("v")
        if vid:
            return vid[0]

    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")


# def ingest_pdf_file(file_content: bytes, filename: str, thread_id: str) -> Dict[str, Any]:
#     """
#     Ingest PDF and related YouTube videos into Qdrant vector store.
#     """
#     doc_id = str(uuid.uuid4())[:8]
#     tmp_path = None

#     try:
#         # 1. Ingest PDF
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
#             f.write(file_content)
#             tmp_path = f.name
        

#         loader = PyPDFLoader(tmp_path)
#         docs = loader.load()
#         splits = splitter.split_documents(docs)
#         splits = [d for d in splits if d.page_content.strip()]

#         if not splits:
#             raise ValueError("No text in PDF")

#         # Store PDF embeddings
#         collection = get_collection_name(thread_id, "pdf", doc_id)
#         pdf_store = get_vector_store(collection)
#         pdf_store.add_documents(splits)
#         vector_stores[collection] = pdf_store

#         doc_info = {
#             "doc_id": doc_id,
#             "filename": filename,
#             "chunks": len(splits),
#             "collection": collection
#         }

#         # if thread_id not in sessions:
#         sessions[thread_id] = {"pdf_docs": []}
#         sessions[thread_id]["pdf_docs"].append(doc_info)
        
      

#         # 2. Search for Context
#         retrieved = pdf_store.similarity_search("what is the topic of pdf", k=5)
#         context = "\n\n".join(d.page_content for d in retrieved)

#         # 🤖 Ask LLM for Topic
#         prompt = f"""
#         Answer the question using only the context below.
#         Context: {context}
#         Question: "topic of the pdf"
#         """
#         response = llm.invoke(prompt)
#         topic = response.content
#         print(f"Topic identified: {topic}")

#         # 3. Fetch Related Videos
#         url = "https://google.serper.dev/videos"
#         payload = {"q": topic}
#         headers = {
#             "X-API-KEY": "1b3dcdb7ee679d34ce513e2a1177db3f81144a32",
#             "Content-Type": "application/json"
#         }

#         res = requests.post(url, json=payload, headers=headers)
#         videos = res.json().get("videos", [])

#         cnt = 0
#         # 4. Ingest Videos
#         for video in videos:
#             print(f"video topis : {video.get('title')}")
#             video_url = video.get("link")
#             if not video_url:
#                 continue

#             video_id = extract_video_id(video_url)
#             if not video_id:
#                 continue

#             try:
#                 transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
#                 print("dsklf;asdfj;a : {transcript}")                
#                 # Combine transcript into one text block or keeping parts based on preference
#                 text_parts = []
#                 for t in transcript:
#                     if isinstance(t, dict):
#                         text_content = t.get("text", "")
#                     else:
#                         text_content = getattr(t, 'text', "")
                    
#                     if text_content:  # Skip empty strings
#                         text_parts.append(str(text_content))
                
#                 text = " ".join(text_parts)


#                 # Create Documents
#                 video_docs = splitter.create_documents([text])

#                 # --- CHANGE START ---
#                 # Use video_id to create a unique collection key
#                 # This effectively "appends" this video's store to your dictionary
                
#                 # Option A: Unique collection per video (Separated Data)
#                 yt_collection_name = f"yt_{video_id}"
                
#                 # Option B: Shared collection (Aggregated Data), but keyed by ID in dict
#                 # yt_collection_name = get_yt_collection() 

#                 yt_store = get_vector_store(yt_collection_name)
#                 yt_store.add_documents(video_docs)

#                 # Store in dictionary using video_id instead of appending to 'success' list
#                 vector_stores[yt_collection_name] = yt_store
                
#                 print(f"✅ Ingested video: {video_id} into collection: {yt_collection_name}")
#                 cnt=cnt+1
#                 # --- CHANGE END 
#                 if(cnt > 2):
#                     break

#             except Exception as e:
#                 print(f"Skipping video {video_id}: {e}")
#                 continue
    
#         return {
#             "doc_id": doc_id,
#             "chunks": len(splits),
#             "total_pdfs": len(sessions[thread_id]["pdf_docs"])
#         }

#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.unlink(tmp_path)


# def ingest_youtube_video(video_id: str, title: Optional[str], thread_id: str) -> Dict[str, Any]:
#     """Ingest a YouTube video transcript"""
#     doc_id = str(uuid.uuid4())[:8]
    
#     try:
#         transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
#     except Exception as e:
#         raise ValueError(f"Failed to fetch YouTube transcript: {str(e)}")
    
#     text_parts = []
#     for t in transcript:
#         text_content = t.get("text", "") if isinstance(t, dict) else getattr(t, 'text', "")
#         if text_content:
#             text_parts.append(str(text_content))
    
#     if not text_parts:
#         raise ValueError("No transcript content found")
    
#     text = " ".join(text_parts)
#     docs = splitter.create_documents([text])
    
#     collection = get_collection_name(thread_id, "yt", doc_id)
#     store = get_vector_store(collection)
#     store.add_documents(docs)
#     vector_stores[collection] = store
    
#     # Store document info
#     doc_info = {
#         "doc_id": doc_id,
#         "video_id": video_id,
#         "title": title or video_id,
#         "chunks": len(docs),
#         "collection": collection
#     }
#     sessions[thread_id]["youtube_docs"].append(doc_info)
    
#     return {
#         "doc_id": doc_id,
#         "chunks": len(docs),
#         "total_videos": len(sessions[thread_id]["youtube_docs"])
#     }

def ingest_pdf_file(file_content: bytes, filename: str, thread_id: str) -> Dict[str, Any]:
    """
    Ingest PDF and related YouTube videos into Qdrant vector store with Metadata.
    """
    doc_id = str(uuid.uuid4())[:8]
    tmp_path = None

    try:
        # --- 1. Ingest PDF ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(file_content)
            tmp_path = f.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Add metadata to PDF chunks
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata["type"] = "pdf"
            doc.metadata["page"] = doc.metadata.get("page", 0) # Ensure page exists

        splits = splitter.split_documents(docs)
        splits = [d for d in splits if d.page_content.strip()]

        if not splits:
            raise ValueError("No text in PDF")

        collection = get_collection_name(thread_id, "pdf", doc_id)
        pdf_store = get_vector_store(collection)
        pdf_store.add_documents(splits)
        vector_stores[collection] = pdf_store

        doc_info = {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(splits),
            "collection": collection
        }

        # Initialize session safely
        if thread_id not in sessions:
            sessions[thread_id] = {"pdf_docs": [], "youtube_docs": []}
        
        # Ensure keys exist if session was created differently elsewhere
        if "pdf_docs" not in sessions[thread_id]:
            sessions[thread_id]["pdf_docs"] = []
        if "youtube_docs" not in sessions[thread_id]:
            sessions[thread_id]["youtube_docs"] = []

        sessions[thread_id]["pdf_docs"].append(doc_info)
        
        # --- 2. Identify Topic ---
        try:
            retrieved = pdf_store.similarity_search("what is the topic of pdf", k=5)
            context = "\n\n".join(d.page_content for d in retrieved)

            prompt = f"""
            Identify the main technical topic of the following text in 5 words or less. 
            Text: {context}
            """
            response = llm.invoke(prompt)
            topic = response.content
            print(f"Topic identified: {topic}")
        except Exception as e:
            print(f"Topic extraction failed: {e}")
            topic = "General Technical Topic" # Fallback

        # --- 3. Fetch Related Videos ---
        try:
            url = "https://google.serper.dev/videos"
            payload = {"q": topic}
            headers = {
                "X-API-KEY": "1b3dcdb7ee679d34ce513e2a1177db3f81144a32",
                "Content-Type": "application/json"
            }

            res = requests.post(url, json=payload, headers=headers)
            videos = res.json().get("videos", [])
        except Exception as e:
            print(f"Serper API failed: {e}")
            videos = []

        cnt = 0
        # --- 4. Ingest Videos ---
        for video in videos:
            try:
                video_url = video.get("link")
                video_title = video.get("title", "Unknown Video")
                
                if not video_url: continue
                video_id = extract_video_id(video_url)
                if not video_id: continue

                # Fetch Transcript
                transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
                
                # Robust text extraction (Handles both dicts and objects)
                text_parts = []
                for t in transcript:
                    if isinstance(t, dict):
                        text_content = t.get("text", "")
                    else:
                        text_content = getattr(t, 'text', "")
                    
                    if text_content:
                        text_parts.append(str(text_content))
                
                text = " ".join(text_parts)
                if not text.strip():
                    print(f"Empty transcript for {video_id}")
                    continue

                # Create Video Documents with Metadata
                video_docs = splitter.create_documents(
                    [text],
                    metadatas=[{
                        "source": video_title,
                        "title": video_title,
                        "url": video_url,
                        "video_id": video_id,
                        "type": "youtube"
                    }]
                )

                yt_collection_name = f"yt_{video_id}"
                yt_store = get_vector_store(yt_collection_name)
                yt_store.add_documents(video_docs)
                vector_stores[yt_collection_name] = yt_store
                
                # Register video in session
                video_info = {
                    "doc_id": str(uuid.uuid4())[:8],
                    "video_id": video_id,
                    "title": video_title,
                    "chunks": len(video_docs),
                    "collection": yt_collection_name
                }
                
                # Safe append
                sessions[thread_id]["youtube_docs"].append(video_info)
                
                print(f"✅ Ingested video: {video_title}")
                cnt += 1
                if cnt >= 2: # Limit to 2 videos
                    break

            except Exception as e:
                print(f"Skipping video {video_id} due to error: {e}")
                continue
    
        return {
            "doc_id": doc_id,
            "chunks": len(splits),
            "total_pdfs": len(sessions[thread_id]["pdf_docs"])
        }

    except Exception as e:
        # Catch-all for the main function to prevent 500 crashes
        print(f"Critical Error in ingest_pdf_file: {e}")
        raise ValueError(f"Ingestion failed: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def ingest_youtube_video(video_id: str, title: Optional[str], thread_id: str) -> Dict[str, Any]:
    """Ingest a YouTube video transcript with Metadata"""
    doc_id = str(uuid.uuid4())[:8]
    
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    except Exception as e:
        raise ValueError(f"Failed to fetch YouTube transcript: {str(e)}")
    
    text_parts = [t.get("text", "") for t in transcript]
    text = " ".join(text_parts)
    
    # ✅ FIX: Add Metadata
    video_title = title or f"YouTube Video {video_id}"
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    docs = splitter.create_documents(
        [text],
        metadatas=[{
            "source": video_title,
            "title": video_title,
            "url": video_url,
            "video_id": video_id,
            "type": "youtube"
        }]
    )
    
    collection = get_collection_name(thread_id, "yt", doc_id)
    store = get_vector_store(collection)
    store.add_documents(docs)
    vector_stores[collection] = store
    
    doc_info = {
        "doc_id": doc_id,
        "video_id": video_id,
        "title": video_title,
        "chunks": len(docs),
        "collection": collection
    }
    
    if thread_id not in sessions:
        sessions[thread_id] = {"pdf_docs": [], "youtube_docs": []}
        
    sessions[thread_id]["youtube_docs"].append(doc_info)
    
    return {
        "doc_id": doc_id,
        "chunks": len(docs),
        "total_videos": len(sessions[thread_id]["youtube_docs"])
    }

current_thread_id: Optional[str] = None

# @tool
# def search_pdf(query: str) -> dict:
#     """
#     Search ALL uploaded PDF documents for relevant information.
#     Use this when the user asks questions about any PDF files they've uploaded.
#     """
#     if current_thread_id is None:
#         return {"content": "No active session."}
    
#     if current_thread_id not in sessions:
#         return {"content": "Session not found."}
    
#     pdf_docs = sessions[current_thread_id].get("pdf_docs", [])
    
#     if not pdf_docs:
#         return {"content": "No PDF documents found in this session."}
    
#     # Get all PDF collections for this session
#     collections = [doc["collection"] for doc in pdf_docs]
    
#     # Search across all PDF collections
#     docs = search_all_collections(collections, query)
    
#     if not docs:
#         return {"content": "No relevant information found in PDFs."}
    
#     context = "\n\n".join(d.page_content for d in docs)
#     return {"content": context}

# @tool
# def search_youtube(query: str) -> dict:
#     """
#     Search ALL YouTube video transcripts for relevant information.
#     Use this when the user asks questions about any YouTube videos they've added.
#     """
#     if current_thread_id is None:
#         return {"content": "No active session."}
    
#     if current_thread_id not in sessions:
#         return {"content": "Session not found."}
    
#     yt_docs = sessions[current_thread_id].get("youtube_docs", [])
    
#     if not yt_docs:
#         return {"content": "No YouTube videos found in this session."}
    
#     # Get all YouTube collections for this session
#     collections = [doc["collection"] for doc in yt_docs]
    
#     # Search across all YouTube collections
#     docs = search_all_collections(collections, query)
    
#     if not docs:
#         return {"content": "No relevant information found in YouTube videos."}
    
#     context = "\n\n".join(d.page_content for d in docs)
#     return {"content": context}


@tool
def search_pdf(query: str) -> dict:
    """
    Search ALL uploaded PDF documents.
    Returns content with citation information (Filename, Page Number).
    """
    if current_thread_id is None or current_thread_id not in sessions:
        return {"content": "No active session."}
    
    pdf_docs = sessions[current_thread_id].get("pdf_docs", [])
    if not pdf_docs:
        return {"content": "No PDF documents found."}
    
    collections = [doc["collection"] for doc in pdf_docs]
    docs = search_all_collections(collections, query)
    
    if not docs:
        return {"content": "No relevant information found in PDFs."}
    
    # ✅ FIX: Format output to include Citation Source
    results = []
    for d in docs:
        source = d.metadata.get("source", "Unknown PDF")
        page = d.metadata.get("page", "N/A")
        # +1 to page because langchain usually uses 0-index
        page_display = int(page) + 1 if str(page).isdigit() else page
        
        results.append(
            f"--- Source: {source} (Page {page_display}) ---\n{d.page_content}\n"
        )

    return {"content": "\n".join(results)}

@tool
def search_youtube(query: str) -> dict:
    """
    Search ALL YouTube video transcripts.
    Returns content with citation information (Video Title, URL).
    """
    if current_thread_id is None or current_thread_id not in sessions:
        return {"content": "No active session."}
    
    yt_docs = sessions[current_thread_id].get("youtube_docs", [])
    
    # Also check for videos that were added via PDF ingestion (stored in vector_stores but maybe not in session list explicitly if logic varies, 
    # but strictly based on your code structure, we rely on collections known to the session)
    
    collections = [doc["collection"] for doc in yt_docs]
    
    # NOTE: If you want to search videos found via PDF ingestion automatically, 
    # you might need to track those collection names in the session object in `ingest_pdf_file`.
    # For now, this searches explicitly added videos.
    
    docs = search_all_collections(collections, query)
    
    if not docs:
        return {"content": "No relevant information found in YouTube videos."}
    
    # ✅ FIX: Format output to include Citation Source
    results = []
    for d in docs:
        title = d.metadata.get("title", "Unknown Video")
        url = d.metadata.get("url", "No URL")
        
        results.append(
            f"--- Source: {title} (URL: {url}) ---\n{d.page_content}\n"
        )
        
    return {"content": "\n".join(results)}

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


search = DuckDuckGoSearchRun()
tools = [search_pdf, search_youtube, recommend_youtube]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_CITATION_PROMPT = """
You are a helpful assistant for a RAG (Retrieval Augmented Generation) system.
You have access to PDF documents and YouTube video transcripts.

Instructions:
1. Answer the user's question based ONLY on the retrieved context.
2. **CITATIONS ARE MANDATORY:** - If the information comes from a PDF, you MUST cite the Filename and Page Number. (e.g., [Source: manual.pdf, Page 5])
   - If the information comes from a YouTube video, you MUST cite the Video Title and provide the Link. (e.g., [Source: "Intro to Python" (https://youtu.be/...)])
3. If the context contains multiple sources, cite all of them.
4. If you cannot find the answer in the provided context, clearly state that you don't know.
"""

class ChatState(TypedDict):
    messages: Annotated[list, operator.add]

def chat_node(state: ChatState):
    # ✅ FIX: Prepend System Message to instructions
    messages = [SystemMessage(content=SYSTEM_CITATION_PROMPT)] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
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

chatbot = graph.compile(checkpointer=InMemorySaver())


@app.post("/session/new", response_model=SessionResponse, tags=["Session"])
async def create_new_session():
    """Create a new chat session"""
    thread_id = create_session()
    return SessionResponse(
        thread_id=thread_id,
        message="New session created successfully",
        status="success"
    )


@app.get("/session/{thread_id}/status", response_model=SessionStatusResponse, tags=["Session"])
async def get_session_status(thread_id: str):
    """Get detailed session status"""
    if thread_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {thread_id} not found")
    
    session = sessions[thread_id]
    pdf_docs = session.get("pdf_docs", [])
    yt_docs = session.get("youtube_docs", [])
    
    # Build PDF document info
    pdf_documents = []
    for doc in pdf_docs:
        pdf_documents.append(DocumentInfo(
            doc_id=doc["doc_id"],
            filename=doc["filename"],
            chunks=doc["chunks"],
            collection_name=doc["collection"]
        ))
    
    youtube_documents = []
    for doc in yt_docs:
        youtube_documents.append(DocumentInfo(
            doc_id=doc["doc_id"],
            filename=doc.get("title", doc.get("video_id", "Unknown")),
            chunks=doc["chunks"],
            collection_name=doc["collection"]
        ))
    
    return SessionStatusResponse(
        thread_id=thread_id,
        total_pdfs=len(pdf_docs),
        total_youtube=len(yt_docs),
        total_pdf_chunks=sum(doc["chunks"] for doc in pdf_docs),
        total_youtube_chunks=sum(doc["chunks"] for doc in yt_docs),
        pdf_documents=pdf_documents,
        youtube_documents=youtube_documents,
        message="Session status retrieved successfully"
    )

# @app.post("/ingest/pdf", response_model=PDFIngestResponse, tags=["Ingestion"])
# async def ingest_pdf(
#     file: UploadFile = File(...),
#     thread_id: Optional[str] = Query(None)
# ):  

#     print(f"file in ingest_pdf : {file}")
#     """Upload and ingest a PDF file (supports multiple PDFs per session)"""
#     if not file.filename.endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="File must be a PDF")
    
#     try:
        
#         file_content = await file.read()
#         if len(file_content) == 0:
#             raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
#         result = ingest_pdf_file(file_content, file.filename, thread_id)
        
#         return PDFIngestResponse(
#             thread_id=thread_id,
#             doc_id=result["doc_id"],
#             filename=file.filename,
#             chunks_created=result["chunks"],
#             total_pdfs=result["total_pdfs"],
#             message=f"PDF '{file.filename}' ingested successfully ({result['total_pdfs']} total PDFs)",
#             status="success"
#         )
    
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

from fastapi import Form, File, UploadFile


@app.post("/ingest/pdf", response_model=PDFIngestResponse, tags=["Ingestion"])
async def ingest_pdf(
    file: UploadFile = File(...),
    thread_id: Optional[str] = Form(None)  # ✅ Change here
):

    print("Thread ID:", thread_id)
    print("File:", file.filename)

    if not file.filename.endswith(".pdf"):
        print("File is not a PDF")
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        if not thread_id:
            print("Thread ID is not provided")
            raise HTTPException(status_code=400, detail="thread_id is required")

        file_content = await file.read()

        if len(file_content) == 0:
            print("File is empty")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        result = ingest_pdf_file(file_content, file.filename, thread_id)

        return PDFIngestResponse(
            thread_id=thread_id,
            doc_id=result["doc_id"],
            filename=file.filename,
            chunks_created=result["chunks"],
            total_pdfs=result["total_pdfs"],
            message=f"PDF '{file.filename}' ingested successfully",
            status="success"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/ingest/youtube", response_model=YouTubeIngestResponse, tags=["Ingestion"])
async def ingest_youtube(
    request: YouTubeIngestRequest,
    thread_id: Optional[str] = Query(None)
):
    """Ingest a YouTube video (supports multiple videos per session)"""
    try:
        if thread_id:
            if thread_id not in sessions:
                raise HTTPException(status_code=404, detail=f"Session {thread_id} not found")
        else:
            thread_id = create_session()
        
        result = ingest_youtube_video(request.video_id, request.title, thread_id)
        
        return YouTubeIngestResponse(
            thread_id=thread_id,
            doc_id=result["doc_id"],
            video_id=request.video_id,
            chunks_created=result["chunks"],
            total_videos=result["total_videos"],
            message=f"YouTube video ingested successfully ({result['total_videos']} total videos)",
            status="success"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Chat with the bot (searches across ALL PDFs and YouTube videos in session)"""
    global current_thread_id

    print(f"requeset in chat : {request}")
    
    # if request.thread_id not in sessions:
    #     raise HTTPException(
    #         status_code=404,
    #         detail=f"Session {request.thread_id} not found"
    #     )

    print(f"session in chat : {sessions}")
    
    try:
        current_thread_id = request.thread_id
        config = {"configurable": {"thread_id": request.thread_id}}
        
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )
        
        last_message = result["messages"][-1]
        
        # ✅ FIX: Handle both string and list content
        if hasattr(last_message, 'content'):
            content = last_message.content
            
            # If content is a list, extract text from it
            if isinstance(content, list):
                response_text = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        response_text += item.get("text", "")
                    elif isinstance(item, str):
                        response_text += item
            else:
                response_text = str(content)
        else:
            response_text = str(last_message)
        
        # Extract tool calls if any
        tool_calls = None
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_calls = [
                {"name": tc.get("name"), "args": tc.get("args", {})}
                for tc in last_message.tool_calls
            ]
        
        return ChatResponse(
            thread_id=request.thread_id,
            response=response_text,
            tool_calls=tool_calls,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        current_thread_id = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)