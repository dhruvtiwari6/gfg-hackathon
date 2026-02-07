# from fastapi import FastAPI, UploadFile, File, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict, Any
# import uuid
# import os
# import tempfile
# import operator
# from typing import TypedDict, Annotated

# from dotenv import load_dotenv
# load_dotenv()

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from youtube_transcript_api import YouTubeTranscriptApi
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
# from langchain.tools import tool
# from langgraph.graph import START, END, StateGraph
# from langgraph.prebuilt import ToolNode
# from langgraph.checkpoint.memory import InMemorySaver
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_core.messages import HumanMessage, AIMessage

# # ============================================================================
# # FASTAPI APP SETUP
# # ============================================================================

# app = FastAPI(
#     title="RAG Chatbot API",
#     description="Multi-session RAG chatbot supporting PDF and YouTube video Q&A",
#     version="1.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # CORS middleware - configure for production
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change to specific origins in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ============================================================================
# # PYDANTIC MODELS
# # ============================================================================

# class SessionResponse(BaseModel):
#     """Response model for session creation"""
#     thread_id: str
#     message: str
#     status: str = "success"

# class PDFIngestResponse(BaseModel):
#     """Response model for PDF ingestion"""
#     thread_id: str
#     chunks_created: int
#     message: str
#     status: str = "success"

# class YouTubeIngestRequest(BaseModel):
#     """Request model for YouTube video ingestion"""
#     video_id: str = Field(..., description="YouTube video ID (e.g., 'dQw4w9WgXcQ')")

# class YouTubeIngestResponse(BaseModel):
#     """Response model for YouTube ingestion"""
#     thread_id: str
#     chunks_created: int
#     message: str
#     status: str = "success"

# class ChatMessage(BaseModel):
#     """Individual chat message"""
#     role: str = Field(..., description="Message role: 'user' or 'assistant'")
#     content: str = Field(..., description="Message content")

# class ChatRequest(BaseModel):
#     """Request model for chat"""
#     message: str = Field(..., description="User message")
#     thread_id: str = Field(..., description="Session thread ID")

# class ChatResponse(BaseModel):
#     """Response model for chat"""
#     thread_id: str
#     response: str
#     tool_calls: Optional[List[Dict[str, Any]]] = None
#     status: str = "success"

# class SessionStatusResponse(BaseModel):
#     """Response model for session status"""
#     thread_id: str
#     has_pdf: bool
#     has_youtube: bool
#     pdf_chunks: int
#     youtube_chunks: int
#     message: str

# class ErrorResponse(BaseModel):
#     """Error response model"""
#     error: str
#     detail: Optional[str] = None
#     status: str = "error"

# # ============================================================================
# # BACKEND LOGIC - SESSION MANAGEMENT
# # ============================================================================

# # Session storage: thread_id -> session data
# sessions: Dict[str, Dict[str, Any]] = {}

# # Vector stores per session
# vector_stores: Dict[str, QdrantVectorStore] = {}

# # Qdrant client
# client = QdrantClient(path="./qdrant_data")

# # Models
# embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# # Splitter
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# def create_session(thread_id: str = None) -> str:
#     """Create a new session or return existing thread ID"""
#     if thread_id is None:
#         thread_id = str(uuid.uuid4())
    
#     if thread_id not in sessions:
#         sessions[thread_id] = {
#             "created_at": None,
#             "pdf_chunks": 0,
#             "youtube_chunks": 0,
#             "has_pdf": False,
#             "has_youtube": False
#         }
    
#     return thread_id

# def get_pdf_collection(thread_id: str) -> str:
#     """Get collection name for PDF documents in session"""
#     return f"pdf_{thread_id}"

# def get_yt_collection(thread_id: str) -> str:
#     """Get collection name for YouTube transcripts in session"""
#     return f"yt_{thread_id}"

# def get_vector_store(collection_name: str) -> QdrantVectorStore:
#     """Get or create a Qdrant vector store for the given collection"""
#     if not client.collection_exists(collection_name):
#         client.create_collection(
#             collection_name=collection_name,
#             vectors_config=VectorParams(
#                 size=1024,
#                 distance=Distance.COSINE
#             )
#         )
    
#     return QdrantVectorStore(
#         client=client,
#         embedding=embeddings,
#         collection_name=collection_name
#     )

# def search_collection(collection: str, query: str, k: int = 4):
#     """Search a collection for relevant documents"""
#     if collection not in vector_stores:
#         if not client.collection_exists(collection):
#             return None
#         vector_stores[collection] = get_vector_store(collection)
    
#     store = vector_stores[collection]
#     return store.similarity_search(query, k=k)

# # ============================================================================
# # INGESTION FUNCTIONS
# # ============================================================================

# def ingest_pdf_file(file_content: bytes, thread_id: str) -> int:
#     """Ingest PDF file into vector store"""
#     tmp_path = None
    
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
        
#         collection = get_pdf_collection(thread_id)
#         store = get_vector_store(collection)
#         store.add_documents(splits)
#         vector_stores[collection] = store
        
#         # Update session info
#         sessions[thread_id]["has_pdf"] = True
#         sessions[thread_id]["pdf_chunks"] = len(splits)
        
#         return len(splits)
    
#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.unlink(tmp_path)

# def ingest_youtube_video(video_id: str, thread_id: str) -> int:
#     """Fetch and ingest YouTube video transcript"""
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
    
#     collection = get_yt_collection(thread_id)
#     store = get_vector_store(collection)
#     store.add_documents(docs)
#     vector_stores[collection] = store
    
#     # Update session info
#     sessions[thread_id]["has_youtube"] = True
#     sessions[thread_id]["youtube_chunks"] = len(docs)
    
#     return len(docs)

# # ============================================================================
# # TOOLS DEFINITION
# # ============================================================================

# # Thread-local storage for current thread_id
# current_thread_id: Optional[str] = None

# @tool
# def search_pdf(query: str) -> dict:
#     """
#     Search uploaded PDF documents for relevant information.
#     Use this tool when the user asks questions about uploaded PDF files.
#     """
#     if current_thread_id is None:
#         return {"content": "No active session."}
    
#     collection = get_pdf_collection(current_thread_id)
#     docs = search_collection(collection, query)
    
#     if not docs:
#         return {"content": "No PDF data found in this session."}
    
#     context = "\n\n".join(d.page_content for d in docs)
#     return {"content": context}

# @tool
# def search_youtube(query: str) -> dict:
#     """
#     Search YouTube video transcripts for relevant information.
#     Use this tool when the user asks questions about YouTube videos they've added.
#     """
#     if current_thread_id is None:
#         return {"content": "No active session."}
    
#     collection = get_yt_collection(current_thread_id)
#     docs = search_collection(collection, query)
    
#     if not docs:
#         return {"content": "No YouTube data found in this session."}
    
#     context = "\n\n".join(d.page_content for d in docs)
#     return {"content": context}

# @tool
# def recommend_youtube(topic: str) -> dict:
#     """Recommend YouTube videos for a given topic."""
#     search = DuckDuckGoSearchRun()
#     query = f"site:youtube.com/watch {topic}"
    
#     try:
#         results = search.run(query)
#         return {
#             "topic": topic,
#             "recommendations": results
#         }
#     except Exception as e:
#         return {
#             "topic": topic,
#             "recommendations": f"Error fetching recommendations: {str(e)}"
#         }

# # Setup tools
# search = DuckDuckGoSearchRun()
# tools = [search_pdf, search_youtube, search, recommend_youtube]
# llm_with_tools = llm.bind_tools(tools)

# # ============================================================================
# # LANGGRAPH SETUP
# # ============================================================================

# class ChatState(TypedDict):
#     messages: Annotated[list, operator.add]

# def chat_node(state: ChatState):
#     """Chat node that invokes the LLM"""
#     response = llm_with_tools.invoke(state["messages"])
#     return {"messages": [response]}

# tool_node = ToolNode(tools)

# def route(state: ChatState):
#     """Route based on whether tools need to be called"""
#     last = state["messages"][-1]
#     if getattr(last, "tool_calls", None):
#         return "tools"
#     return "end"

# # Build graph
# graph = StateGraph(ChatState)
# graph.add_node("chat", chat_node)
# graph.add_node("tools", tool_node)
# graph.add_edge(START, "chat")
# graph.add_conditional_edges("chat", route, {
#     "tools": "tools",
#     "end": END
# })
# graph.add_edge("tools", "chat")

# chatbot = graph.compile(checkpointer=InMemorySaver())

# # ============================================================================
# # API ENDPOINTS
# # ============================================================================

# @app.get("/", tags=["Health"])
# async def root():
#     """Health check and API information"""
#     return {
#         "service": "RAG Chatbot API",
#         "version": "1.0.0",
#         "status": "running",
#         "endpoints": {
#             "GET /": "API information",
#             "POST /session/new": "Create new chat session",
#             "GET /session/{thread_id}/status": "Get session status",
#             "POST /ingest/pdf": "Upload and ingest PDF",
#             "POST /ingest/youtube": "Ingest YouTube video",
#             "POST /chat": "Send chat message",
#             "GET /health": "Health check"
#         },
#         "documentation": {
#             "swagger": "/docs",
#             "redoc": "/redoc"
#         }
#     }

# @app.get("/health", tags=["Health"])
# async def health_check():
#     """Detailed health check"""
#     return {
#         "status": "healthy",
#         "services": {
#             "qdrant": "connected" if client else "disconnected",
#             "embeddings": "loaded" if embeddings else "not loaded",
#             "llm": "loaded" if llm else "not loaded"
#         },
#         "active_sessions": len(sessions),
#         "vector_collections": len(vector_stores)
#     }

# @app.post("/session/new", response_model=SessionResponse, tags=["Session"])
# async def create_new_session():
#     """
#     Create a new chat session.
    
#     Returns a unique thread_id to use for subsequent requests.
#     """
#     thread_id = create_session()
    
#     return SessionResponse(
#         thread_id=thread_id,
#         message="New session created successfully",
#         status="success"
#     )

# @app.get("/session/{thread_id}/status", response_model=SessionStatusResponse, tags=["Session"])
# async def get_session_status(thread_id: str):
#     """
#     Get status of a specific session.
    
#     - **thread_id**: Session thread ID
#     """
#     if thread_id not in sessions:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Session {thread_id} not found"
#         )
    
#     session = sessions[thread_id]
    
#     return SessionStatusResponse(
#         thread_id=thread_id,
#         has_pdf=session.get("has_pdf", False),
#         has_youtube=session.get("has_youtube", False),
#         pdf_chunks=session.get("pdf_chunks", 0),
#         youtube_chunks=session.get("youtube_chunks", 0),
#         message="Session status retrieved successfully"
#     )

# @app.post("/ingest/pdf", response_model=PDFIngestResponse, tags=["Ingestion"])
# async def ingest_pdf(
#     file: UploadFile = File(..., description="PDF file to upload"),
#     thread_id: Optional[str] = Query(None, description="Session thread ID (creates new if not provided)")
# ):
#     """
#     Upload and ingest a PDF file.
    
#     - **file**: PDF file to upload
#     - **thread_id**: Optional session thread ID (creates new session if not provided)
    
#     The PDF will be split into chunks and stored in a vector database for retrieval.
#     """
#     # Validate file type
#     if not file.filename.endswith('.pdf'):
#         raise HTTPException(
#             status_code=400,
#             detail="File must be a PDF (.pdf extension required)"
#         )
    
#     try:
#         # Create or get session
#         if thread_id:
#             if thread_id not in sessions:
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"Session {thread_id} not found"
#                 )
#         else:
#             thread_id = create_session()
        
#         # Read file content
#         file_content = await file.read()
        
#         if len(file_content) == 0:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Uploaded file is empty"
#             )
        
#         # Ingest PDF
#         chunks_created = ingest_pdf_file(file_content, thread_id)
        
#         return PDFIngestResponse(
#             thread_id=thread_id,
#             chunks_created=chunks_created,
#             message=f"PDF '{file.filename}' ingested successfully with {chunks_created} chunks",
#             status="success"
#         )
    
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error ingesting PDF: {str(e)}"
#         )

# @app.post("/ingest/youtube", response_model=YouTubeIngestResponse, tags=["Ingestion"])
# async def ingest_youtube(
#     request: YouTubeIngestRequest,
#     thread_id: Optional[str] = Query(None, description="Session thread ID (creates new if not provided)")
# ):
#     """
#     Ingest a YouTube video transcript.
    
#     - **video_id**: YouTube video ID (e.g., 'dQw4w9WgXcQ' from https://youtube.com/watch?v=dQw4w9WgXcQ)
#     - **thread_id**: Optional session thread ID (creates new session if not provided)
    
#     The transcript will be fetched and stored in a vector database for retrieval.
#     """
#     try:
#         # Create or get session
#         if thread_id:
#             if thread_id not in sessions:
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"Session {thread_id} not found"
#                 )
#         else:
#             thread_id = create_session()
        
#         # Ingest YouTube video
#         chunks_created = ingest_youtube_video(request.video_id, thread_id)
        
#         return YouTubeIngestResponse(
#             thread_id=thread_id,
#             chunks_created=chunks_created,
#             message=f"YouTube video '{request.video_id}' ingested successfully with {chunks_created} chunks",
#             status="success"
#         )
    
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error ingesting YouTube video: {str(e)}"
#         )

# @app.post("/chat", response_model=ChatResponse, tags=["Chat"])
# async def chat(request: ChatRequest):
#     """
#     Send a chat message and get a response.
    
#     - **message**: User message
#     - **thread_id**: Session thread ID (required - create session first)
    
#     The chatbot will use available tools (PDF search, YouTube search, web search) 
#     to answer questions based on ingested content.
#     """
#     global current_thread_id
    
#     # Validate session
#     if request.thread_id not in sessions:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Session {request.thread_id} not found. Create a session first using /session/new"
#         )
    
#     try:
#         # Set current thread for tools
#         current_thread_id = request.thread_id
        
#         # Prepare config
#         config = {"configurable": {"thread_id": request.thread_id}}
        
#         # Invoke chatbot
#         result = chatbot.invoke(
#             {"messages": [HumanMessage(content=request.message)]},
#             config=config
#         )
        
#         # Extract response
#         last_message = result["messages"][-1]
#         response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
#         # Extract tool calls if any
#         tool_calls = None
#         if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#             tool_calls = [
#                 {
#                     "name": tc.get("name"),
#                     "args": tc.get("args", {})
#                 }
#                 for tc in last_message.tool_calls
#             ]
        
#         return ChatResponse(
#             thread_id=request.thread_id,
#             response=response_text,
#             tool_calls=tool_calls,
#             status="success"
#         )
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing chat: {str(e)}"
#         )
#     finally:
#         # Clear current thread
#         current_thread_id = None

# # ============================================================================
# # ERROR HANDLERS
# # ============================================================================

# @app.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     """Custom HTTP exception handler"""
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={
#             "error": exc.detail,
#             "status": "error",
#             "status_code": exc.status_code
#         }
#     )

# @app.exception_handler(Exception)
# async def general_exception_handler(request, exc):
#     """General exception handler"""
#     return JSONResponse(
#         status_code=500,
#         content={
#             "error": "Internal server error",
#             "detail": str(exc),
#             "status": "error",
#             "status_code": 500
#         }
#     )

# # ============================================================================
# # STARTUP/SHUTDOWN EVENTS
# # ============================================================================

# @app.on_event("startup")
# async def startup_event():
#     """Initialize services on startup"""
#     print("=" * 70)
#     print("🚀 RAG Chatbot API Starting...")
#     print("=" * 70)
#     print(f"✓ Qdrant client initialized")
#     print(f"✓ Embeddings model loaded: nvidia/nv-embedqa-e5-v5")
#     print(f"✓ LLM loaded: gemini-2.5-flash-lite")
#     print(f"✓ LangGraph chatbot compiled")
#     print("=" * 70)
#     print("📚 API Documentation available at:")
#     print("   - Swagger UI: http://localhost:8000/docs")
#     print("   - ReDoc: http://localhost:8000/redoc")
#     print("=" * 70)

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Cleanup on shutdown"""
#     print("\n🛑 Shutting down RAG Chatbot API...")
#     # Add any cleanup code here if needed

# # ============================================================================
# # MAIN
# # ============================================================================

# if __name__ == "__main__":
#     import uvicorn
    
#     uvicorn.run(
#         "fastapi_backend:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )




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
from langchain_core.messages import HumanMessage, AIMessage

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
    allow_origins=["*"],
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

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

# Session storage structure:
# {
#   "thread_id": {
#     "created_at": timestamp,
#     "pdf_docs": [
#       {"doc_id": "uuid", "filename": "doc.pdf", "chunks": 42, "collection": "pdf_thread_docid"}
#     ],
#     "youtube_docs": [
#       {"doc_id": "uuid", "video_id": "xyz", "chunks": 15, "collection": "yt_thread_docid"}
#     ]
#   }
# }

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

# ============================================================================
# INGESTION FUNCTIONS
# ============================================================================

def ingest_pdf_file(file_content: bytes, filename: str, thread_id: str) -> Dict[str, Any]:
    """Ingest a single PDF file"""
    doc_id = str(uuid.uuid4())[:8]  # Short ID
    tmp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(file_content)
            tmp_path = f.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splits = splitter.split_documents(docs)
        splits = [d for d in splits if d.page_content.strip()]
        
        if not splits:
            raise ValueError("No text content found in PDF")
        
        collection = get_collection_name(thread_id, "pdf", doc_id)
        store = get_vector_store(collection)
        store.add_documents(splits)
        vector_stores[collection] = store
        
        # Store document info
        doc_info = {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(splits),
            "collection": collection
        }
        sessions[thread_id]["pdf_docs"].append(doc_info)
        
        return {
            "doc_id": doc_id,
            "chunks": len(splits),
            "total_pdfs": len(sessions[thread_id]["pdf_docs"])
        }
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def ingest_youtube_video(video_id: str, title: Optional[str], thread_id: str) -> Dict[str, Any]:
    """Ingest a YouTube video transcript"""
    doc_id = str(uuid.uuid4())[:8]
    
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    except Exception as e:
        raise ValueError(f"Failed to fetch YouTube transcript: {str(e)}")
    
    text_parts = []
    for t in transcript:
        text_content = t.get("text", "") if isinstance(t, dict) else getattr(t, 'text', "")
        if text_content:
            text_parts.append(str(text_content))
    
    if not text_parts:
        raise ValueError("No transcript content found")
    
    text = " ".join(text_parts)
    docs = splitter.create_documents([text])
    
    collection = get_collection_name(thread_id, "yt", doc_id)
    store = get_vector_store(collection)
    store.add_documents(docs)
    vector_stores[collection] = store
    
    # Store document info
    doc_info = {
        "doc_id": doc_id,
        "video_id": video_id,
        "title": title or video_id,
        "chunks": len(docs),
        "collection": collection
    }
    sessions[thread_id]["youtube_docs"].append(doc_info)
    
    return {
        "doc_id": doc_id,
        "chunks": len(docs),
        "total_videos": len(sessions[thread_id]["youtube_docs"])
    }

# ============================================================================
# TOOLS DEFINITION
# ============================================================================

current_thread_id: Optional[str] = None

@tool
def search_pdf(query: str) -> dict:
    """
    Search ALL uploaded PDF documents for relevant information.
    Use this when the user asks questions about any PDF files they've uploaded.
    """
    if current_thread_id is None:
        return {"content": "No active session."}
    
    if current_thread_id not in sessions:
        return {"content": "Session not found."}
    
    pdf_docs = sessions[current_thread_id].get("pdf_docs", [])
    
    if not pdf_docs:
        return {"content": "No PDF documents found in this session."}
    
    # Get all PDF collections for this session
    collections = [doc["collection"] for doc in pdf_docs]
    
    # Search across all PDF collections
    docs = search_all_collections(collections, query)
    
    if not docs:
        return {"content": "No relevant information found in PDFs."}
    
    context = "\n\n".join(d.page_content for d in docs)
    return {"content": context}

@tool
def search_youtube(query: str) -> dict:
    """
    Search ALL YouTube video transcripts for relevant information.
    Use this when the user asks questions about any YouTube videos they've added.
    """
    if current_thread_id is None:
        return {"content": "No active session."}
    
    if current_thread_id not in sessions:
        return {"content": "Session not found."}
    
    yt_docs = sessions[current_thread_id].get("youtube_docs", [])
    
    if not yt_docs:
        return {"content": "No YouTube videos found in this session."}
    
    # Get all YouTube collections for this session
    collections = [doc["collection"] for doc in yt_docs]
    
    # Search across all YouTube collections
    docs = search_all_collections(collections, query)
    
    if not docs:
        return {"content": "No relevant information found in YouTube videos."}
    
    context = "\n\n".join(d.page_content for d in docs)
    return {"content": context}

@tool
def recommend_youtube(topic: str) -> dict:
    """Recommend YouTube videos for a given topic."""
    search = DuckDuckGoSearchRun()
    query = f"site:youtube.com/watch {topic}"
    
    try:
        results = search.run(query)
        return {"topic": topic, "recommendations": results}
    except Exception as e:
        return {"topic": topic, "recommendations": f"Error: {str(e)}"}

search = DuckDuckGoSearchRun()
tools = [search_pdf, search_youtube, search, recommend_youtube]
llm_with_tools = llm.bind_tools(tools)

# ============================================================================
# LANGGRAPH SETUP
# ============================================================================

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

chatbot = graph.compile(checkpointer=InMemorySaver())

# ============================================================================
# API ENDPOINTS
# ============================================================================

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
    
    # Build YouTube document info
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

# @app.get("/session/{thread_id}/status", response_model=SessionStatusResponse, tags=["Session"])
# async def get_session_status(thread_id: str):
#     """Get detailed session status"""
#     if thread_id not in sessions:
#         raise HTTPException(status_code=404, detail=f"Session {thread_id} not found")
    
#     session = sessions[thread_id]
#     pdf_docs = session.get("pdf_docs", [])
#     yt_docs = session.get("youtube_docs", [])
    
#     return SessionStatusResponse(
#         thread_id=thread_id,
#         total_pdfs=len(pdf_docs),
#         total_youtube=len(yt_docs),
#         total_pdf_chunks=sum(doc["chunks"] for doc in pdf_docs),
#         total_youtube_chunks=sum(doc["chunks"] for doc in yt_docs),
#         pdf_documents=[DocumentInfo(**doc) for doc in pdf_docs],
#         youtube_documents=[
#             DocumentInfo(
#                 doc_id=doc["doc_id"],
#                 filename=doc.get("title", doc["video_id"]),
#                 chunks=doc["chunks"],
#                 collection_name=doc["collection"]
#             ) for doc in yt_docs
#         ],
#         message="Session status retrieved successfully"
    # )

@app.post("/ingest/pdf", response_model=PDFIngestResponse, tags=["Ingestion"])
async def ingest_pdf(
    file: UploadFile = File(...),
    thread_id: Optional[str] = Query(None)
):
    """Upload and ingest a PDF file (supports multiple PDFs per session)"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        if thread_id:
            if thread_id not in sessions:
                raise HTTPException(status_code=404, detail=f"Session {thread_id} not found")
        else:
            thread_id = create_session()
        
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        result = ingest_pdf_file(file_content, file.filename, thread_id)
        
        return PDFIngestResponse(
            thread_id=thread_id,
            doc_id=result["doc_id"],
            filename=file.filename,
            chunks_created=result["chunks"],
            total_pdfs=result["total_pdfs"],
            message=f"PDF '{file.filename}' ingested successfully ({result['total_pdfs']} total PDFs)",
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

# @app.post("/chat", response_model=ChatResponse, tags=["Chat"])
# async def chat(request: ChatRequest):
#     """Chat with the bot (searches across ALL PDFs and YouTube videos in session)"""
#     global current_thread_id
    
#     if request.thread_id not in sessions:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Session {request.thread_id} not found"
#         )
    
#     try:
#         current_thread_id = request.thread_id
#         config = {"configurable": {"thread_id": request.thread_id}}
        
#         result = chatbot.invoke(
#             {"messages": [HumanMessage(content=request.message)]},
#             config=config
#         )
        
#         last_message = result["messages"][-1]
#         response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
#         tool_calls = None
#         if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#             tool_calls = [
#                 {"name": tc.get("name"), "args": tc.get("args", {})}
#                 for tc in last_message.tool_calls
#             ]
        
#         return ChatResponse(
#             thread_id=request.thread_id,
#             response=response_text,
#             tool_calls=tool_calls,
#             status="success"
#         )
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
#     finally:
#         current_thread_id = None


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Chat with the bot (searches across ALL PDFs and YouTube videos in session)"""
    global current_thread_id
    
    if request.thread_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {request.thread_id} not found"
        )
    
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

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)