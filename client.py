# import asyncio
# import sys
# import os
# from langchain_mcp_adapters.client import MultiServerMCPClient

# async def main():
#     # 1. Initialize the client (No "await" or "async with" here)
#     client = MultiServerMCPClient(
#         {
#             "manim": {
#                 "transport": "stdio",
#                 "command": sys.executable,
#                 "args": ["main.py"] 
#             }
#         }
#     )

#     # 2. This is the only line that needs "await" to start the servers
#     tools = await client.get_tools()

#     print(f"Available tools: {[t.name for t in tools]}")

#     tool_name = "execute_manim_code"
#     tool = next((t for t in tools if t.name == tool_name), None)

#     if tool:
#         prompt = """
# from manim import *
# class Demo(Scene):
#     def construct(self):
#         self.play(Write(Text("Hello World")))
# """
#         # 3. Invoke the tool
#         result = await tool.ainvoke({"manim_code": prompt})
#         print(result)
#     else:
#         print(f"Error: {tool_name} not found.")

# if __name__ == "__main__":
#     asyncio.run(main())


# working above ****************

# import asyncio
# import sys
# import os
# import operator
# from typing import TypedDict, Annotated
# from dotenv import load_dotenv

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.graph import StateGraph, START
# from langchain_core.messages import HumanMessage, SystemMessage

# load_dotenv()

# class AgentState(TypedDict):
#     messages: Annotated[list, operator.add]

# async def main():
#     model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

#     client = MultiServerMCPClient(
#         {
#             "manim_server": {
#                 "transport": "stdio",
#                 "command": sys.executable,
#                 "args": ["main.py"] 
#             }
#         }
#     )

#     mcp_tools = await client.get_tools()
#     model_with_tools = model.bind_tools(mcp_tools)

#     async def call_model(state: AgentState):
#         system_prompt = SystemMessage(content=(
#             "You are a video production robot. You ONLY use tools.\n"
#             "Step 1: Use 'execute_manim_code'.\n"
#             "Step 2: Provide the final path to the user."
#         ))
#         response = await model_with_tools.ainvoke([system_prompt] + state["messages"])
#         return {"messages": [response]}

#     workflow = StateGraph(AgentState)
#     workflow.add_node("agent", call_model)
#     workflow.add_node("tools", ToolNode(mcp_tools))
#     workflow.add_edge(START, "agent")
#     workflow.add_conditional_edges("agent", tools_condition)
#     workflow.add_edge("tools", "agent")
#     app = workflow.compile()

#     inputs = {"messages": [HumanMessage(content="Create a 30 sec Manim video explaining Classification.")]}
    
#     print("--- Starting Agent ---")
#     async for chunk in app.astream(inputs, stream_mode="values"):
#         if not chunk["messages"]: continue
#         last_message = chunk["messages"][-1]
        
#         # FIX: Check if tool_calls exists AND is not empty
#         if hasattr(last_message, "tool_calls") and last_message.tool_calls:
#             print(f"\n[ACTION] Calling Tool: {last_message.tool_calls[0]['name']}")
        
#         elif hasattr(last_message, "content") and last_message.content:
#             # Check if it's a list (Tool output) or string (AI message)
#             content = last_message.content
#             if isinstance(content, list):
#                 print(f"\n[TOOL RESULT] {content[0].get('text')}")
#             elif not isinstance(last_message, HumanMessage):
#                 print(f"\n[AGENT] {content}")

# if __name__ == "__main__":
#     asyncio.run(main())



# working above ****************
import asyncio
import sys
import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

async def main():
    # Gemini 2.0 Flash is recommended for tool use
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    # Connect to the server defined in main.py
    client = MultiServerMCPClient(
        {
            "manim_server": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["main.py"] 
            }
        }
    )

    mcp_tools = await client.get_tools()
    model_with_tools = model.bind_tools(mcp_tools)

    async def call_model(state: AgentState):
        system_prompt = SystemMessage(content=(
            "You are an expert Manim video production assistant.\n\n"
            "ALWAYS use 'create_video_with_narration' for requests involving voiceovers.\n"
            "The tool will automatically handle video/audio duration sync.\n\n"
            "MANIM RULES:\n"
            "- Use 'from manim import *'\n"
            "- Always use 3D coordinates: [x, y, 0]\n"
            "- When creating a Decision Boundary, use a Line or a FunctionGraph.\n"
            "- Use VGroup(*[mobs]) (unpack the list).\n"
            "- Keep animations concise. The tool adds a final wait to match narration."
        ))
        response = await model_with_tools.ainvoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(mcp_tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    app = workflow.compile()

    inputs = {
        "messages": [
            HumanMessage(content=(
                "Create a Manim video explaining Classification. "
                "Show blue dots on the left, red dots on the right, and a "
                "yellow decision boundary line appearing between them with narration."
            ))
        ]
    }

    print("--- Starting Video Production ---")
    async for chunk in app.astream(inputs, stream_mode="values"):
        if not chunk["messages"]: continue
        last_msg = chunk["messages"][-1]
        
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            print(f"\n[ACTION] Calling: {last_msg.tool_calls[0]['name']}")
        elif hasattr(last_msg, "content") and last_msg.content:
            if not isinstance(last_msg, HumanMessage):
                print(f"\n[RESULT] {last_msg.content}")

if __name__ == "__main__":
    asyncio.run(main())