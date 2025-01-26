from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from expert import Agent_expert, PydanticAIDeps
from expert import *

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
USER_ICON = "icons\\user_icon.png"  
BOT_ICON = "icons\\bot_icon.png"  
# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


async def run_agent_with_streaming(Agent_expert, user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    # Run the agent in a stream
    async with Agent_expert.expert.run_stream(
        user_input,
        deps=deps,
        message_history= st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

async def save_chat_to_supabase(role: str, content: str, source: str):
    """
    Save a chat message to Supabase.
    """
    from datetime import datetime
    timestamp = datetime.utcnow().isoformat()
    chat_message = {
        "role": role,
        "timestamp": timestamp,
        "content": content, 
        "source": source
    }
    
    try:
        response = supabase.table("chat_messages").insert(chat_message).execute()
    except Exception as e:
        print(f"Error saving chat to database: {e}")
        st.error("Failed to save chat")


async def load_chats_from_supabase(source: str):
    """
    Retrieve all chat messages from Supabase and update session state.
    """
    try:
        response = supabase.table("chat_messages").select("role, timestamp, content").eq("source", source).order("timestamp", desc=False).execute()
        return response.data
    except Exception as e:
        print(f"Error loading chats from database: {e}")
        st.error("Failed to load chats from database")

def display_chat(chat_data):
    # Create containers for User, LLM, and System messages
    user_container = st.container()
    model_container = st.container()
    
    # Loop through the chat data and display messages accordingly
    for message in chat_data:
        role = message['role']
        content = message['content']
        
        if role == 'user':
            with st.chat_message(role, avatar=USER_ICON):
                st.markdown(content)
        elif role == 'model':
            with st.chat_message(role, avatar=BOT_ICON):
                st.markdown(content)
        else:
            # You can handle the system's role or any additional roles here
            with user_container:
                st.markdown(f"**System**: {content}")


async def main():
    if "to_init" not in st.session_state:
        st.session_state.to_init = False
        llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
        model = OpenAIModel(llm)
        st.session_state.agent = Agent_expert("pyspark", model = model, doc_source= "pydantic_ai_docs")
        st.title(f"{st.session_state.agent.agent_scope} AI Agentic RAG")
        st.write(f"Ask any question about {st.session_state.agent.agent_scope}, the hidden truths of the beauty of this framework lie within.")
        st.session_state.previous_chats = await load_chats_from_supabase(source="pyspark")
        

    agent = st.session_state.agent
    display_chat(st.session_state.previous_chats)
    


    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input(f"What questions do you have about {agent.agent_scope}?", key="user_input")

    if user_input:
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        await save_chat_to_supabase("user", user_input, source=agent.agent_scope)

        with st.chat_message("user", avatar=USER_ICON):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=BOT_ICON):
            await run_agent_with_streaming(agent, user_input)

            # Save assistant response to Supabase
            last_message = st.session_state.messages[-1]
            if isinstance(last_message, ModelResponse):
                assistant_response = " ".join(part.content for part in last_message.parts if isinstance(part, TextPart))
                await save_chat_to_supabase("model", assistant_response, source=agent.agent_scope)


def run():
    asyncio.run(main())

if __name__ == "__main__":
    run()
