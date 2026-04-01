from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

system_prompt = """

You are a personal chef. You will be given an image of leftover ingredients in the fridge.

First List the ingredients you see in the image.

Then, using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("meta-llama/llama-4-scout-17b-16e-instruct", model_provider="groq")

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
)
