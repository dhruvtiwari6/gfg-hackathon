
# import streamlit as st
# import uuid

# from backend import chatbot, load_pdf, set_current_thread_id
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.runnables import RunnableConfig


# # Helpers

# def generate_thread_id():
#     return str(uuid.uuid4())

# def normalize_content(content):
#     """
#     Safely normalize AIMessage.content into a string.

#     Handles:
#     - str
#     - list[str]
#     - list[{"type": "text", "text": "..."}]  (Gemini)
#     - mixed lists (rare but possible)
#     - empty / tool-only messages
#     """
#     if content is None:
#         return ""

#     # Case 1: plain string
#     if isinstance(content, str):
#         return content

#     # Case 2: list (Gemini / streaming)
#     if isinstance(content, list):
#         parts = []
#         for part in content:
#             if isinstance(part, dict):
#                 if part.get("type") == "text":
#                     parts.append(part.get("text", ""))
#             elif isinstance(part, str):
#                 parts.append(part)
#         return "".join(parts)

#     return str(content)


# def add_thread(thread_id):
#     set_current_thread_id(thread_id)
#     if thread_id not in st.session_state["all_thread"]:
#         st.session_state["all_thread"].append(thread_id)


# def new_chat():
#     thread_id = generate_thread_id()
#     st.session_state["thread_id"] = thread_id
#     st.session_state["message_history"] = []

#     add_thread(thread_id)


# def load_chat(thread_id):
#     """
#     Load messages from LangGraph state
#     and normalize them for UI rendering
#     """
#     messages = chatbot.get_state(
#         config={"configurable": {"thread_id": thread_id}}
#     ).values.get("messages", [])

#     temp = []
#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             temp.append({"role": "user", "content": msg.content})

#         elif isinstance(msg, AIMessage):
#             content = normalize_content(msg.content)
#             if content:  # ignore tool-only steps
#                 temp.append({"role": "assistant", "content": content})

#     return temp


# # Session Setup

# if "message_history" not in st.session_state:
#     st.session_state["message_history"] = []

# if "thread_id" not in st.session_state:
#     st.session_state["thread_id"] = generate_thread_id()

# if "all_thread" not in st.session_state:
#     st.session_state["all_thread"] = []

# add_thread(st.session_state["thread_id"])


# config: RunnableConfig = {
#     "configurable": {"thread_id": st.session_state["thread_id"]}
# }

# # Sidebar

# st.sidebar.title("LangGraph Bot")

# if st.sidebar.button("➕ New Chat"):
#     new_chat()

# st.sidebar.header("Chats")

# for tid in st.session_state["all_thread"]:
#     if st.sidebar.button(str(tid)):
#         set_current_thread_id(tid)
#         st.session_state["thread_id"] = tid
#         config["configurable"]["thread_id"] = tid
#         st.session_state["message_history"] = load_chat(tid)

# # PDF Upload in Sidebar
# st.sidebar.header("Upload PDF")
# uploaded_file = st.sidebar.file_uploader(
#     "Choose a PDF file",
#     type=["pdf"],
#     key="pdf_uploader"
# )

# if uploaded_file is not None:
#     # Check if this file was already processed
#     if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
        
#         # Add to chat history
#         with st.chat_message("user"):
#             st.markdown(f"📄 **Uploaded PDF:** {uploaded_file.name}")

#         st.session_state["message_history"].append({
#             "role": "user",
#             "content": f"Uploaded PDF: {uploaded_file.name}",
#             "type": "pdf",
#             "filename": uploaded_file.name
#         })

#         # Process the file
#         with st.sidebar.status("Processing file..."):
#             load_pdf(uploaded_file)
#             st.sidebar.success("✓ File processed successfully!")
        
#         # Mark this file as processed
#         st.session_state["last_uploaded_file"] = uploaded_file.name

# # Render Chat History

# for msg in st.session_state["message_history"]:
#     with st.chat_message(msg["role"]):
#         if msg.get("type") == "pdf":
#             st.markdown(
#                 f"📄 **Uploaded PDF:** {msg.get('filename', 'document.pdf')}"
#             )
#         else:
#             st.markdown(msg["content"])

# # Text Input

# user_input = st.chat_input("Say something")

# # Text Message Handling (Streaming)

# if user_input:
#     query = user_input

#     # ---- User Message ----
#     with st.chat_message("user"):
#         st.markdown(query)

#     st.session_state["message_history"].append(
#         {"role": "user", "content": query}
#     )

#     # ---- Assistant (Streaming) ----
#     with st.chat_message("assistant"):
#         ai_msg = st.write_stream(
#             (
#                 normalize_content(chunk.content)
#                 for chunk, _ in chatbot.stream(
#                     {"messages": [HumanMessage(content=query)]},
#                     config=config,
#                     stream_mode="messages",
#                 )
#                 if isinstance(chunk, AIMessage)
#                 and normalize_content(chunk.content)
#             )
#         )

#     # ---- Save Assistant Message ----
#     st.session_state["message_history"].append(
#         {"role": "assistant", "content": ai_msg}
#     )


import streamlit as st
import uuid

from streamlit_backend import chatbot, ingest_pdf, ingest_youtube
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig


# Helpers

def generate_thread_id():
    return str(uuid.uuid4())

def normalize_content(content):
    """
    Safely normalize AIMessage.content into a string.

    Handles:
    - str
    - list[str]
    - list[{"type": "text", "text": "..."}]  (Gemini)
    - mixed lists (rare but possible)
    - empty / tool-only messages
    """
    if content is None:
        return ""

    # Case 1: plain string
    if isinstance(content, str):
        return content

    # Case 2: list (Gemini / streaming)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)

    return str(content)


def add_thread(thread_id):
    if thread_id not in st.session_state["all_thread"]:
        st.session_state["all_thread"].append(thread_id)


def new_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def load_chat(thread_id):
    """
    Load messages from LangGraph state
    and normalize them for UI rendering
    """
    messages = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    ).values.get("messages", [])

    temp = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp.append({"role": "user", "content": msg.content})

        elif isinstance(msg, AIMessage):
            content = normalize_content(msg.content)
            if content:  # ignore tool-only steps
                temp.append({"role": "assistant", "content": content})

    return temp


# Session Setup

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "all_thread" not in st.session_state:
    st.session_state["all_thread"] = []

add_thread(st.session_state["thread_id"])


config: RunnableConfig = {
    "configurable": {"thread_id": st.session_state["thread_id"]}
}

# Sidebar

st.sidebar.title("LangGraph Bot")

if st.sidebar.button("➕ New Chat"):
    new_chat()

st.sidebar.header("Chats")

for tid in st.session_state["all_thread"]:
    if st.sidebar.button(str(tid)):
        st.session_state["thread_id"] = tid
        config["configurable"]["thread_id"] = tid
        st.session_state["message_history"] = load_chat(tid)

# PDF Upload in Sidebar
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    key="pdf_uploader"
)

if uploaded_file is not None:
    # Check if this file was already processed
    if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
        
        # Add to chat history
        with st.chat_message("user"):
            st.markdown(f"📄 **Uploaded PDF:** {uploaded_file.name}")

        st.session_state["message_history"].append({
            "role": "user",
            "content": f"Uploaded PDF: {uploaded_file.name}",
            "type": "pdf",
            "filename": uploaded_file.name
        })

        # Process the file using correct backend function
        with st.sidebar.status("Processing file..."):
            num_chunks = ingest_pdf(uploaded_file)
            st.sidebar.success(f"✓ File processed! {num_chunks} chunks created.")
        
        # Mark this file as processed
        st.session_state["last_uploaded_file"] = uploaded_file.name

# YouTube Upload in Sidebar
st.sidebar.header("Add YouTube Video")
youtube_url = st.sidebar.text_input(
    "Enter YouTube URL or Video ID",
    key="youtube_input",
    placeholder="https://youtube.com/watch?v=... or video_id"
)

if st.sidebar.button("Process YouTube Video"):
    if youtube_url:
        print(f"youtube video url is {youtube_url}")
        # Extract video ID from URL or use as-is
        video_id = youtube_url
        if "youtube.com/watch?v=" in youtube_url:
            video_id = youtube_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
        
        # Add to chat history

        print(f"youtube video id is {video_id}")
        with st.chat_message("user"):
            st.markdown(f"🎥 **Added YouTube Video:** {video_id}")

        st.session_state["message_history"].append({
            "role": "user",
            "content": f"Added YouTube Video: {video_id}",
            "type": "youtube",
            "video_id": video_id
        })

        # Process the video
        try:
            with st.sidebar.status("Processing video transcript..."):
                num_chunks = ingest_youtube(video_id)
                st.sidebar.success(f"✓ Video processed! {num_chunks} chunks created.")
        except Exception as e:
            st.sidebar.error(f"Error processing video: {str(e)}")

# Render Chat History

for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "pdf":
            st.markdown(
                f"📄 **Uploaded PDF:** {msg.get('filename', 'document.pdf')}"
            )
        elif msg.get("type") == "youtube":
            st.markdown(
                f"🎥 **Added YouTube Video:** {msg.get('video_id', 'video')}"
            )
        else:
            st.markdown(msg["content"])

# Text Input

user_input = st.chat_input("Say something")

# Text Message Handling (Streaming)

if user_input:
    query = user_input

    # ---- User Message ----
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state["message_history"].append(
        {"role": "user", "content": query}
    )

    # ---- Assistant (Streaming) ----
    with st.chat_message("assistant"):
        ai_msg = st.write_stream(
            (
                normalize_content(chunk.content)
                for chunk, _ in chatbot.stream(
                    {"messages": [HumanMessage(content=query)]},
                    config=config,
                    stream_mode="messages",
                )
                if isinstance(chunk, AIMessage)
                and normalize_content(chunk.content)
            )
        )

    # ---- Save Assistant Message ----
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_msg}
    )