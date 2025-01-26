import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client
import aiohttp

class WebCrawler:
    def __init__(self, source: str):
        load_dotenv()
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        self.source = source
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
        )

    @dataclass
    class ProcessedChunk:
        url: str
        chunk_number: int
        title: str
        summary: str
        content: str
        metadata: Dict[str, Any]
        embedding: List[float]

    async def get_docs_urls(self, sitemap_url: str) -> List[str]:
        async with aiohttp.ClientSession() as session:
            async with session.get(sitemap_url) as response:
                response_text = await response.text()
                root = ElementTree.fromstring(response_text)
                namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
                return urls

    def chunk_text(self, text: str, chunk_size: int = 5000) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            chunk = text[start:end]
            code_block = chunk.rfind('```')
            if code_block != -1 and code_block > chunk_size * 0.3:
                end = start + code_block
            elif '\n\n' in chunk:
                last_break = chunk.rfind('\n\n')
                if last_break > chunk_size * 0.3:
                    end = start + last_break
            elif '. ' in chunk:
                last_period = chunk.rfind('. ')
                if last_period > chunk_size * 0.3:
                    end = start + last_period + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = max(start + 1, end)
        return chunks

    async def get_title_and_summary(self, chunk: str, url: str) -> Dict[str, str]:
        system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
        Return a JSON object with 'title' and 'summary' keys."""
        try:
            response = await self.openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error getting title and summary: {e}")
            return {"title": "Error", "summary": "Error"}

    async def get_embedding(self, text: str) -> List[float]:
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0] * 1536

    async def process_chunk(self, chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
        extracted = await self.get_title_and_summary(chunk, url)
        embedding = await self.get_embedding(chunk)
        metadata = {
            "source": self.source,
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path
        }
        return self.ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=extracted['title'],
            summary=extracted['summary'],
            content=chunk,
            metadata=metadata,
            embedding=embedding
        )

    async def insert_chunk(self, chunk: ProcessedChunk):
        try:
            data = {
                "url": chunk.url,
                "chunk_number": chunk.chunk_number,
                "title": chunk.title,
                "summary": chunk.summary,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding
            }
            result = self.supabase.table("site_pages").insert(data).execute()
            print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
            return result
        except Exception as e:
            print(f"Error inserting chunk: {e}")
            return None

    async def process_and_store_document(self, url: str, markdown: str):
        chunks = self.chunk_text(markdown)
        semaphore = asyncio.Semaphore(5)
        async def process_chunk_with_limit(chunk: str, chunk_number: int, url: str):
            async with semaphore:
                return await self.process_chunk(chunk, chunk_number, url)
        processed_chunks = await asyncio.gather(
            *(process_chunk_with_limit(chunk, i, url) for i, chunk in enumerate(chunks))
        )
        await asyncio.gather(*(self.insert_chunk(chunk) for chunk in processed_chunks))

    async def main(self, sitemap_url: str):
        urls = await self.get_docs_urls(sitemap_url)
        if urls:
            await self.process_and_store_document(urls[0], "Sample content")

if __name__ == "__main__":
    crawler = WebCrawler("google_cloud")
    asyncio.run(crawler.main("https://cloud.google.com/sitemap.xml"))
