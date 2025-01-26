from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

class Agent_expert:
    def __init__(self, agent_scope: str, model: OpenAIModel, doc_source: str):
        self.agent_scope = agent_scope

        self.system_prompt = f"""
        You are an expert at {self.agent_scope}. 
        You have access to all the documentation, including examples, API references, and other resources to help you build and optimize {self.agent_scope} applications.

        Your only job is to assist with {self.agent_scope}. You don't answer other questions besides describing what you are able to do.

        Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

        When you first look at the documentation, always start with RAG (Retrieval-Augmented Generation).
        Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

        Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
        """

        self.doc_source = doc_source
        self.expert = Agent(
            model,
            system_prompt=self.system_prompt,
            deps_type=PydanticAIDeps,
            retries=2
        )

        # Register tools
        self.expert.tool(self.retrieve_relevant_documentation)
        self.expert.tool(self.list_documentation_pages)
        self.expert.tool(self.get_page_content)


    async def get_embedding(self, text: str, openai_client: AsyncOpenAI) -> List[float]:
        """Get embedding vector from OpenAI."""
        try:
            response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0] * 1536  # Return zero vector on error

    async def retrieve_relevant_documentation(self, ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
        """
        Retrieve relevant documentation chunks based on the query with RAG.
        """
        try:
            query_embedding = await self.get_embedding(user_query, ctx.deps.openai_client)
            result = ctx.deps.supabase.rpc(
                'match_site_pages',
                {
                    'query_embedding': query_embedding,
                    'match_count': 5,
                    'filter': {'source': self.doc_source}
                }
            ).execute()
            if not result.data:
                return "No relevant documentation found."

            formatted_chunks = [f"\n# {doc['title']}\n\n{doc['content']}" for doc in result.data]
            return "\n\n---\n\n".join(formatted_chunks)
        except Exception as e:
            print(f"Error retrieving documentation: {e}")
            return f"Error retrieving documentation: {str(e)}"

    async def list_documentation_pages(self, ctx: RunContext[PydanticAIDeps]) -> List[str]:
        """
        Retrieve a list of all available Pydantic AI documentation pages.
        """
        try:
            result = ctx.deps.supabase.from_('site_pages').select('url').eq('metadata->>source', self.doc_source).execute()
            if not result.data:
                return []
            return sorted(set(doc['url'] for doc in result.data))
        except Exception as e:
            print(f"Error retrieving documentation pages: {e}")
            return []

    async def get_page_content(self, ctx: RunContext[PydanticAIDeps], url: str) -> str:
        """
        Retrieve the full content of a specific documentation page by combining all its chunks.
        """
        try:
            result = ctx.deps.supabase.from_('site_pages').select('title, content, chunk_number').eq('url', url).eq('metadata->>source', self.doc_source).order('chunk_number').execute()
            if not result.data:
                return f"No content found for URL: {url}"
            page_title = result.data[0]['title'].split(' - ')[0]
            formatted_content = [f"# {page_title}\n"] + [chunk['content'] for chunk in result.data]
            return "\n\n".join(formatted_content)
        except Exception as e:
            print(f"Error retrieving page content: {e}")
            return f"Error retrieving page content: {str(e)}"
