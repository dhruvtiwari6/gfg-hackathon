# import os
# import subprocess
# import shutil
# import tempfile
# import re
# from mcp.server.fastmcp import FastMCP

# # Initialize MCP
# mcp = FastMCP("ManimServer")

# MANIM_EXECUTABLE = os.getenv("MANIM_EXECUTABLE", "manim")
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "media"))
# os.makedirs(BASE_DIR, exist_ok=True)

# def get_scene_name(code: str):
#     # Improved regex to catch various Scene types (e.g., VectorScene, MovingCameraScene)
#     match = re.search(r"class\s+(\w+)\s*\(\s*\w*Scene\s*\)", code)
#     return match.group(1) if match else None

# @mcp.tool()
# def execute_manim_code(manim_code: str) -> str:
#     """Execute the Manim code and return the path to the video."""
    
#     # Create a unique temp directory inside BASE_DIR
#     tmpdir = tempfile.mkdtemp(dir=BASE_DIR)
#     script_path = os.path.join(tmpdir, "scene.py")

#     try:
#         scene_name = get_scene_name(manim_code)
#         if not scene_name:
#             return "Error: No Scene class found in the provided code."

#         with open(script_path, "w") as f:
#             f.write(manim_code)

#         # -pql: Preview, Quality Low
#         # --media_dir .: Save output in the current directory to avoid nested media/ folders
#         result = subprocess.run(
#             [MANIM_EXECUTABLE, "-ql", "--media_dir", ".", script_path, scene_name],
#             capture_output=True,
#             text=True,
#             cwd=tmpdir
#         )

#         if result.returncode == 0:
#             # Find the generated mp4 file
#             video_path = ""
#             for root, dirs, files in os.walk(tmpdir):
#                 for file in files:
#                     if file.endswith(".mp4"):
#                         video_path = os.path.join(root, file)
            
#             return f"Success. Video generated at: {video_path}"
#         else:
#             return f"Manim error:\n{result.stderr}\n{result.stdout}"

#     except Exception as e:
#         return f"Execution error: {str(e)}"

# @mcp.tool()
# def cleanup_manim_temp_dir(directory: str) -> str:
#     """Safely delete a temp directory."""
#     try:
#         # Security check: Ensure we are only deleting within BASE_DIR
#         target_dir = os.path.abspath(directory)
#         if not target_dir.startswith(BASE_DIR) or target_dir == BASE_DIR:
#             return "Cleanup denied: Directory is outside the allowed temporary path."
        
#         shutil.rmtree(target_dir)
#         return f"Cleanup successful: {target_dir} removed."
#     except Exception as e:
#         return f"Cleanup failed: {e}"

# if __name__ == "__main__":
#     mcp.run()


#working .... 


# import os
# import re
# import subprocess
# import shutil
# import glob
# from mcp.server.fastmcp import FastMCP

# mcp = FastMCP("ManimServer")
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "media"))
# os.makedirs(BASE_DIR, exist_ok=True)

# def get_scene_name(code: str):
#     match = re.search(r"class\s+(\w+)\s*\(\s*Scene\s*\)", code)
#     return match.group(1) if match else None

# @mcp.tool()
# def execute_manim_code(manim_code: str) -> str:
#     """Executes Manim code. Returns path to the FINAL .mp4 video."""
#     tmpdir = os.path.join(BASE_DIR, "manim_tmp")
#     os.makedirs(tmpdir, exist_ok=True)
#     script_path = os.path.join(tmpdir, "scene.py")
    
#     clean_code = manim_code.replace("\\n", "\n").replace("\\t", "\t")
#     clean_code = clean_code.strip().strip("`").replace("python\n", "")

#     with open(script_path, "w") as f:
#         f.write(clean_code)
    
#     scene = get_scene_name(clean_code)
#     if not scene: return "Error: No Scene class found."

#     result = subprocess.run(
#         ["manim", "-pql", "--media_dir", tmpdir, script_path, scene],
#         capture_output=True, text=True
#     )

#     if result.returncode == 0:
#         # We look for the video, excluding the 'partial_movie_files' directory
#         search_pattern = os.path.join(tmpdir, "videos", "**", f"{scene}.mp4")
#         all_videos = glob.glob(search_pattern, recursive=True)
        
#         # Filter out partial files
#         final_videos = [v for v in all_videos if "partial_movie_files" not in v]
        
#         if final_videos:
#             final_path = os.path.join(BASE_DIR, f"{scene}_final.mp4")
#             shutil.move(final_videos[0], final_path)
#             return f"Success! Video is at: {final_path}"
            
#     return f"Manim Error: {result.stderr}"

# if __name__ == "__main__":
#     mcp.run(transport="stdio")


# import os
# import re
# import subprocess
# import shutil
# import glob
# from mcp.server.fastmcp import FastMCP

# mcp = FastMCP("ManimServer")

# # Setup Directories
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "media"))
# os.makedirs(BASE_DIR, exist_ok=True)

# def get_scene_name(code: str):
#     """Extracts the class name of the Scene."""
#     match = re.search(r"class\s+(\w+)\(Scene\):", code)
#     return match.group(1) if match else "Demo"

# def get_audio_duration(file_path: str) -> float:
#     """Returns the duration of an audio file in seconds using ffprobe."""
#     try:
#         result = subprocess.run(
#             ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
#              "-of", "default=noprint_wrappers=1:nokey=1", file_path],
#             capture_output=True, text=True
#         )
#         return float(result.stdout.strip())
#     except Exception:
#         return 10.0  # Default fallback

# @mcp.tool()
# def execute_manim_code(manim_code: str) -> str:
#     """Executes Manim code and returns the path to the final mp4."""
#     tmpdir = os.path.join(BASE_DIR, "manim_tmp")
#     os.makedirs(tmpdir, exist_ok=True)
#     script_path = os.path.join(tmpdir, "scene.py")

#     try:
#         # Clean up code formatting
#         clean_code = manim_code.replace("\\n", "\n").replace("\\t", "\t")
#         clean_code = clean_code.strip().strip("`").replace("python\n", "")

#         with open(script_path, "w") as f:
#             f.write(clean_code)
        
#         scene = get_scene_name(clean_code)

#         # Run Manim (Low Quality -l for speed)
#         result = subprocess.run(
#             ["manim", "-ql", "--media_dir", tmpdir, script_path, scene],
#             capture_output=True, text=True
#         )

#         if result.returncode == 0:
#             search_pattern = os.path.join(tmpdir, "videos", "**", f"{scene}.mp4")
#             all_videos = glob.glob(search_pattern, recursive=True)
#             final_videos = [v for v in all_videos if "partial_movie_files" not in v]
            
#             if final_videos:
#                 final_path = os.path.join(BASE_DIR, f"{scene}_video.mp4")
#                 shutil.move(final_videos[0], final_path)
#                 return f"Success! Video is at: {final_path}"
#             return "Error: Video generated but file not found."
#         return f"Manim Error: {result.stderr}"
#     except Exception as e:
#         return f"Error: {str(e)}"

# @mcp.tool()
# def generate_audio_narration(text: str, output_filename: str = "narration.mp3") -> str:
#     """Generate TTS audio from text."""
#     try:
#         from gtts import gTTS
#         audio_dir = os.path.join(BASE_DIR, "audio")
#         os.makedirs(audio_dir, exist_ok=True)
#         output_path = os.path.join(audio_dir, output_filename)
        
#         tts = gTTS(text=text, lang='en')
#         tts.save(output_path)
#         return f"Success! Audio generated at: {output_path}"
#     except Exception as e:
#         return f"Error: {str(e)}"

# @mcp.tool()
# def merge_video_audio(video_path: str, audio_path: str, output_filename: str = "final_output.mp4") -> str:
#     """Merges video and audio using FFmpeg."""
#     output_path = os.path.join(BASE_DIR, output_filename)
#     try:
#         # We use -shortest to ensure they match, but the sync logic in 
#         # create_video_with_narration makes this smoother.
#         subprocess.run([
#             "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
#             "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
#             "-shortest", output_path
#         ], capture_output=True, text=True)
#         return f"Success! Final video at: {output_path}"
#     except Exception as e:
#         return f"Error merging: {str(e)}"

# @mcp.tool()
# def create_video_with_narration(manim_code: str, narration_text: str, output_name: str = "ml_vid") -> str:
#     """One-shot tool that syncs Manim animation length to narration length."""
    
#     # 1. Generate Audio first to know the length
#     audio_res = generate_audio_narration(narration_text, f"{output_name}.mp3")
#     if "Error" in audio_res: return audio_res
#     audio_path = audio_res.split("at: ")[-1].strip()
#     duration = get_audio_duration(audio_path)

#     # 2. Inject a wait at the end of the Manim code to match audio length
#     # This prevents the video from ending before the speech finishes.
#     clean_code = manim_code.strip().strip("`").replace("python\n", "")
#     # Find the last indented line and append a wait
#     synced_code = clean_code + f"\n        self.wait({duration})"

#     # 3. Create Video
#     video_res = execute_manim_code(synced_code)
#     if "Error" in video_res: return video_res
#     video_path = video_res.split("at: ")[-1].strip()

#     # 4. Merge
#     return merge_video_audio(video_path, audio_path, f"{output_name}_final.mp4")

# if __name__ == "__main__":
#     mcp.run(transport="stdio")



# import os
# import re
# import subprocess
# import shutil
# import glob
# from mcp.server.fastmcp import FastMCP

# mcp = FastMCP("ManimServer")

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "media"))
# os.makedirs(BASE_DIR, exist_ok=True)

# def get_duration(file_path: str) -> float:
#     """Gets duration of any media file using ffprobe."""
#     try:
#         cmd = [
#             "ffprobe", "-v", "error", "-show_entries", "format=duration",
#             "-of", "default=noprint_wrappers=1:nokey=1", file_path
#         ]
#         result = subprocess.run(cmd, capture_output=True, text=True)
#         return float(result.stdout.strip())
#     except:
#         return 0.0

# @mcp.tool()
# def execute_manim_code(manim_code: str) -> str:
#     """Executes Manim code. Returns path to the final video."""
#     tmpdir = os.path.join(BASE_DIR, "manim_tmp")
#     os.makedirs(tmpdir, exist_ok=True)
#     script_path = os.path.join(tmpdir, "scene.py")

#     clean_code = manim_code.replace("\\n", "\n").replace("\\t", "\t").strip().strip("`").replace("python\n", "")
#     with open(script_path, "w") as f:
#         f.write(clean_code)
    
#     match = re.search(r"class\s+(\w+)\(Scene\):", clean_code)
#     scene = match.group(1) if match else "Demo"

#     result = subprocess.run(
#         ["manim", "-ql", "--media_dir", tmpdir, script_path, scene],
#         capture_output=True, text=True
#     )

#     if result.returncode == 0:
#         videos = glob.glob(os.path.join(tmpdir, "videos", "**", f"{scene}.mp4"), recursive=True)
#         final_videos = [v for v in videos if "partial_movie_files" not in v]
#         if final_videos:
#             dest = os.path.join(BASE_DIR, f"{scene}_temp.mp4")
#             shutil.move(final_videos[0], dest)
#             return dest
#     return f"Error: {result.stderr}"

# @mcp.tool()
# def create_video_with_narration(manim_code: str, narration_text: str, output_name: str = "final") -> str:
#     """
#     Creates video and audio, then syncs them perfectly by padding the shorter one.
#     """
#     # 1. Generate Video
#     video_path = execute_manim_code(manim_code)
#     if "Error" in video_path: return video_path

#     # 2. Generate Audio
#     from gtts import gTTS
#     audio_path = os.path.join(BASE_DIR, f"{output_name}.mp3")
#     gTTS(text=narration_text, lang='en').save(audio_path)

#     # 3. Calculate Durations
#     v_dur = get_duration(video_path)
#     a_dur = get_duration(audio_path)
    
#     final_output = os.path.join(BASE_DIR, f"{output_name}_synced.mp4")

#     # 4. Perfect Sync Logic using FFmpeg Complex Filter
#     # If video is shorter: we pad with last frame (tpad)
#     # If audio is shorter: we pad with silence (adelay/amix)
    
#     if v_dur < a_dur:
#         # Video is shorter: Freeze the last frame
#         diff = a_dur - v_dur
#         cmd = [
#             "ffmpeg", "-y",
#             "-i", video_path,
#             "-i", audio_path,
#             "-vf", f"tpad=stop_mode=clone:stop_duration={diff}",
#             "-c:v", "libx264", "-c:a", "aac", "-shortest",
#             final_output
#         ]
#     else:
#         # Audio is shorter: Just merge (it will have silence at the end)
#         cmd = [
#             "ffmpeg", "-y",
#             "-i", video_path,
#             "-i", audio_path,
#             "-c:v", "copy", "-c:a", "aac",
#             "-map", "0:v:0", "-map", "1:a:0",
#             final_output
#         ]

#     subprocess.run(cmd, capture_output=True)
#     return f"Success! Video created: {final_output}"

# if __name__ == "__main__":
#     mcp.run(transport="stdio")

import os
import re
import subprocess
import shutil
import glob
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ManimServer")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "media"))
os.makedirs(BASE_DIR, exist_ok=True)

@mcp.tool()
def create_video_with_narration(manim_code: str, narration_text: str, output_name: str = "final") -> str:
    """
    Simplified Linux Merge: Robust and explicit.
    """
    # 1. Manim Generation
    tmpdir = os.path.join(BASE_DIR, "manim_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    script_path = os.path.join(tmpdir, "scene.py")
    
    clean_code = manim_code.replace("\\n", "\n").replace("\\t", "\t").strip().strip("`").replace("python\n", "")
    with open(script_path, "w") as f:
        f.write(clean_code)
    
    match = re.search(r"class\s+(\w+)\(Scene\):", clean_code)
    scene = match.group(1) if match else "Demo"

    print(f"Generating Manim Scene: {scene}")
    subprocess.run(["manim", "-ql", "--media_dir", tmpdir, script_path, scene], capture_output=True)
    
    video_files = glob.glob(os.path.join(tmpdir, "videos", "**", f"{scene}.mp4"), recursive=True)
    video_path = [v for v in video_files if "partial_movie_files" not in v][0]

    # 2. Audio Generation
    from gtts import gTTS
    audio_path = os.path.join(BASE_DIR, f"{output_name}.mp3")
    gTTS(text=narration_text, lang='en').save(audio_path)

    # 3. THE BRUTE FORCE MERGE
    # This command uses the '-shortest' flag and the 'shortest_buf' fix.
    # It will ensure the file is created with BOTH audio and video.
    final_output = os.path.join(BASE_DIR, f"{output_name}_final.mp4")
    
    # We use -fflags +genpts to fix timing issues common in Manim outputs
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-pix_fmt", "yuv420p",
        "-shortest", # Cuts off at the end of the shorter one (usually video)
        final_output
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return f"ERROR during FFmpeg: {result.stderr}"

    return f"SUCCESS: Video at {final_output}"

if __name__ == "__main__":
    mcp.run(transport="stdio")