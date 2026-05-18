# tools/research_tools.py
import os
import requests

from typing import Type
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from crewai.tools import BaseTool
from tavily import TavilyClient


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# TAVILY SEARCH INPUT SCHEMA
# =========================================================

class TavilySearchInput(BaseModel):
    """
    Input schema for Tavily Search Tool.
    """

    query: str = Field(
        ...,
        description="Search query used to retrieve information from the internet."
    )


# =========================================================
# TAVILY SEARCH TOOL
# =========================================================

class TavilySearchTool(BaseTool):
    """
    Custom CrewAI tool for Tavily AI-powered search.
    """

    name: str = "Tavily Search Tool"

    description: str = (
        "Useful for searching recent internet information, "
        "scientific research, articles, discoveries, "
        "news, and detailed web knowledge."
    )

    args_schema: Type[BaseModel] = TavilySearchInput

    def __init__(self):
        """
        Initialize Tavily client once.
        """

        super().__init__()

        # IMPORTANT:
        # Use _client instead of client
        # because CrewAI tools use Pydantic internally
        self._client = TavilyClient(
            api_key=os.getenv("TAVILY_API_KEY")
        )

    def _run(self, query: str) -> str:
        """
        Execute Tavily search.
        """

        try:

            response = self._client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )

            return str(response)

        except Exception as e:
            return f"Tavily Search Failed: {str(e)}"


# =========================================================
# JINA READER INPUT SCHEMA
# =========================================================

class JinaReaderInput(BaseModel):
    """
    Input schema for Jina Reader Tool.
    """

    url: str = Field(
        ...,
        description="Website URL to extract readable webpage content from."
    )


# =========================================================
# JINA READER TOOL
# =========================================================

class JinaReaderTool(BaseTool):
    """
    Custom CrewAI tool for webpage extraction
    using Jina AI Reader.
    """

    name: str = "Jina Reader Tool"

    description: str = (
        "Useful for extracting clean readable webpage "
        "content from articles, blogs, research papers, "
        "documentation, and websites."
    )

    args_schema: Type[BaseModel] = JinaReaderInput

    def _run(self, url: str) -> str:
        """
        Extract webpage content using Jina Reader.
        """

        try:

            # Remove protocol for Jina formatting
            clean_url = (
                url.replace("https://", "")
                .replace("http://", "")
            )

            # Create Jina Reader URL
            jina_url = f"https://r.jina.ai/http://{clean_url}"

            # Fetch webpage content
            response = requests.get(
                jina_url,
                timeout=20
            )

            # Raise error if request failed
            response.raise_for_status()

            return response.text

        except Exception as e:
            return f"Jina Reader Failed: {str(e)}"