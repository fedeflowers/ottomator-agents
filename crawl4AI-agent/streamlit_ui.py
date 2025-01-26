from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI
import re
from urllib.parse import urlparse

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
from crawl_docs import WebCrawler
from crawl_docs import *

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

async def save_domain_to_supabase(domain: str):
    """
    Save domain to Supabase.
    """
    msg = {
        "doc_type": domain,
    }
    
    try:
        response = supabase.table("documentation").insert(msg).execute()
    except Exception as e:
        print(f"Error saving domain to database: {e}")
        st.error("Failed to save domain")


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

def retrieve_docs_list(collection):
    try:
        res = []
        response = supabase.table(collection).select("doc_type").execute()
        for el in response.data:
            res.append(el["doc_type"])
        return res
    except Exception as e:
        print(f"Error loading games from database: {e}")
        st.error("Failed to load docs from database")

def is_valid_sitemap(url):
    """
    Validate if the given URL follows the correct sitemap format: 
    'https://<domain>/sitemap.xml'
    """
    pattern = r"^https:\/\/[\w.-]+\/sitemap\.xml$"
    return re.match(pattern, url) is not None
def extract_domain_from_sitemap(url):
    """
    Extracts the domain from a given sitemap URL.

    Example:
    Input:  'https://spark.apache.org/sitemap.xml'
    Output: 'spark.apache.org'
    """
    parsed_url = urlparse(url)
    return parsed_url.netloc


async def crawl_and_store(url):
    try:
        if is_valid_sitemap(url):
            st.session_state.site_map = url
            st.sidebar.success(f"Valid sitemap URL submitted: {url}")

            # Extract domain and initialize WebCrawler
            domain = extract_domain_from_sitemap(url)
            st.session_state.crawler = WebCrawler(domain)

            # Save domain asynchronously to Supabase
            await save_domain_to_supabase(domain)

            # Start crawling the sitemap URL
            await st.session_state.crawler.main(url)

            st.sidebar.success("Crawling completed successfully!")
        else:
            st.sidebar.error("Invalid sitemap format. Use: https://example.com/sitemap.xml")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")
    
async def main():
    if "to_init" not in st.session_state:
        st.session_state.docs = "documentation"
        st.session_state.to_init = False
        llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
        st.session_state.model = OpenAIModel(llm)
        st.session_state.crawler = WebCrawler("pyspark")
        st.session_state.site_map = "https://spark.apache.org/sitemap.xml"
        #if no docs found, crawl a doc to start
        try:
            st.session_state.selected_doc = retrieve_docs_list(st.session_state.docs)[-1]
        except:
            pass
        
    # SIDEBAR PREVIOUS CONVERSATIONS
    doc_options = retrieve_docs_list(st.session_state.docs)

    # Store the selected dpc in session state
    if len(doc_options) != 0 :
        st.sidebar.title("Previous Conversations")
    else:
        st.sidebar.markdown("No docs found, <br> crawl a dococumentation to start", unsafe_allow_html=True)

    if "selected_doc" in st.session_state:
        selected_doc = st.sidebar.selectbox("Select a doc:", doc_options, index=doc_options.index(st.session_state.selected_doc))
        #instantiate agent
        #change pydantic_ai_docs to the selected doc
        st.session_state.agent = Agent_expert(selected_doc, model = st.session_state.model, doc_source= "pydantic_ai_docs")
        st.session_state.previous_chats = await load_chats_from_supabase(source=selected_doc)
        st.title(f"{st.session_state.agent.agent_scope} AI Agentic RAG")
        st.write(f"Ask any question about {st.session_state.agent.agent_scope}, the hidden truths of the beauty of this framework lie within.")

    if "previous_chats" in st.session_state:
        display_chat(st.session_state.previous_chats)


    # Update selected doc only when the selection changes
    if "selected_doc" in st.session_state:
        if selected_doc != st.session_state.selected_doc:
            st.session_state.selected_doc = selected_doc
            st.rerun()


    st.sidebar.divider()
    # # crawl doc button
    st.sidebar.header("Crawl Document")
    url = st.sidebar.text_input("Enter URL to crawl:", "")

    if st.sidebar.button("Crawl Doc"):
        if url:
            asyncio.create_task(crawl_and_store(url))  # Run the async task without blocking UI
        else:
            st.sidebar.warning("Please enter a valid URL.")

    # # Initialize chat history in session state if not present
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
    if "agent" in st.session_state:
        user_input = st.chat_input(f"What questions do you have about {st.session_state.agent.agent_scope}?", key="user_input")

        if user_input:
            st.session_state.messages.append(
                ModelRequest(parts=[UserPromptPart(content=user_input)])
            )
            await save_chat_to_supabase("user", user_input, source=st.session_state.agent.agent_scope)

            with st.chat_message("user", avatar=USER_ICON):
                st.markdown(user_input)

            with st.chat_message("assistant", avatar=BOT_ICON):
                await run_agent_with_streaming(st.session_state.agent, user_input)

                # Save assistant response to Supabase
                last_message = st.session_state.messages[-1]
                if isinstance(last_message, ModelResponse):
                    assistant_response = " ".join(part.content for part in last_message.parts if isinstance(part, TextPart))
                    await save_chat_to_supabase("model", assistant_response, source=st.session_state.agent.agent_scope)


def run():
    asyncio.run(main())

if __name__ == "__main__":
    run()
