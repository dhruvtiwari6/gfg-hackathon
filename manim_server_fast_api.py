from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import asyncio
import json
from typing import Optional
import subprocess
import uuid

app = FastAPI(title="Manim Video Generation API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory for outputs
BASE_DIR = os.path.join(os.getcwd(), "media")
os.makedirs(BASE_DIR, exist_ok=True)

# Pydantic models
class SimpleVideoRequest(BaseModel):
    manim_code: str
    output_name: Optional[str] = None

class NarrationVideoRequest(BaseModel):
    manim_code: str
    narration_text: str
    output_name: Optional[str] = None

class VideoResponse(BaseModel):
    status: str
    message: str
    video_path: Optional[str] = None
    video_id: Optional[str] = None

# Helper function to call MCP server
async def call_mcp_tool(tool_name: str, params: dict) -> str:
    """Call an MCP tool via stdio communication"""
    try:
        # Create the MCP request
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            }
        }
        
        # Start the MCP server process
        process = await asyncio.create_subprocess_exec(
            "python", "mcp_server.py",  # Update with your MCP server file path
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Send request
        request_json = json.dumps(request) + "\n"
        stdout, stderr = await process.communicate(request_json.encode())
        
        if process.returncode != 0:
            return f"ERROR: MCP server failed: {stderr.decode()}"
        
        # Parse response
        response = json.loads(stdout.decode().strip())
        if "result" in response:
            return response["result"]["content"][0]["text"]
        elif "error" in response:
            return f"ERROR: {response['error']}"
        else:
            return "ERROR: Unexpected response format"
            
    except Exception as e:
        return f"ERROR: {str(e)}"

@app.get("/")
async def root():
    return {
        "message": "Manim Video Generation API",
        "endpoints": {
            "POST /generate/simple": "Generate simple Manim video",
            "POST /generate/narration": "Generate video with narration",
            "GET /video/{video_id}": "Download generated video",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "output_dir": BASE_DIR}

@app.post("/generate/simple", response_model=VideoResponse)
async def generate_simple_video(request: SimpleVideoRequest):
    """Generate a simple Manim video without narration"""
    try:
        # Generate unique output name if not provided
        output_name = request.output_name or f"video_{uuid.uuid4().hex[:8]}"
        
        # Call MCP tool
        result = await call_mcp_tool(
            "create_simple_manim_video",
            {
                "manim_code": request.manim_code,
                "output_name": output_name
            }
        )
        
        if "ERROR" in result or "Error" in result:
            raise HTTPException(status_code=500, detail=result)
        
        # Extract video path from result
        video_path = result.split("created at ")[-1].strip() if "SUCCESS" in result else None
        
        return VideoResponse(
            status="success",
            message="Video generated successfully",
            video_path=video_path,
            video_id=output_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/narration", response_model=VideoResponse)
async def generate_video_with_narration(request: NarrationVideoRequest):
    """Generate a Manim video with voiceover narration"""
    try:
        # Generate unique output name if not provided
        output_name = request.output_name or f"video_{uuid.uuid4().hex[:8]}"
        
        # Call MCP tool
        result = await call_mcp_tool(
            "create_video_with_narration",
            {
                "manim_code": request.manim_code,
                "narration_text": request.narration_text,
                "output_name": output_name
            }
        )
        
        if "ERROR" in result or "Error" in result:
            raise HTTPException(status_code=500, detail=result)
        
        # Extract video path from result
        video_path = result.split("created: ")[-1].strip() if "Success" in result else None
        
        return VideoResponse(
            status="success",
            message="Video with narration generated successfully",
            video_path=video_path,
            video_id=output_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/{video_id}")
async def download_video(video_id: str):
    """Download a generated video by ID"""
    try:
        # Try different possible filenames
        possible_files = [
            os.path.join(BASE_DIR, f"{video_id}_synced.mp4"),
            os.path.join(BASE_DIR, f"{video_id}_final.mp4"),
            os.path.join(BASE_DIR, f"{video_id}.mp4")
        ]
        
        video_path = None
        for path in possible_files:
            if os.path.exists(path):
                video_path = path
                break
        
        if not video_path:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"{video_id}.mp4"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/videos")
async def list_videos():
    """List all generated videos"""
    try:
        videos = []
        for file in os.listdir(BASE_DIR):
            if file.endswith(".mp4"):
                file_path = os.path.join(BASE_DIR, file)
                videos.append({
                    "filename": file,
                    "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
                    "created": os.path.getctime(file_path)
                })
        
        return {"videos": videos, "count": len(videos)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/video/{video_id}")
async def delete_video(video_id: str):
    """Delete a generated video"""
    try:
        # Try to find and delete the video
        deleted = []
        for pattern in [f"{video_id}_synced.mp4", f"{video_id}_final.mp4", f"{video_id}.mp4"]:
            path = os.path.join(BASE_DIR, pattern)
            if os.path.exists(path):
                os.remove(path)
                deleted.append(pattern)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return {"status": "success", "deleted": deleted}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)