'''
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
'''
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat

#from autogen_agentchat.register import register_function  # if this is where it is in your version
#from autogen_ext.utils import config_list_openai_aoai 
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import os # Necessary for Tavily API key and dummy configuration

# --- Dummy Search Function (Requires real implementation or Tavily setup) ---
# NOTE: To run this code successfully with real search, you need:
# 1. pip install tavily-python requests beautifulsoup4
# 2. Set the TAVILY_API_KEY environment variable.
#    For testing purposes, we'll include a mocked version of `cached_search`.

# If you have a real Tavily setup, uncomment and replace this mock:
# from tavily import TavilyClient
# tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", "YOUR_TAVILY_KEY"))
# def cached_search(query, num_results):
#     return tavily.search(query=query, search_depth="advanced", max_results=num_results)



def web_search(query: str) -> str:
    """
    Performs a web search using Tavily API to find relevant information.
    
    Args:
        query (str): The search query.

    Returns:
        str: JSON-formatted search results (list of objects with title, source, content).
    """
    print(f"\n[TOOL]: Performing web search for: '{query}'")

    try:
        # Call the search implementation
        results = cached_search(query, num_results=3)

        output = []
        for r in results.get("results", []):
            output.append({
                "title": r.get("title", ""),
                "source": r.get("url", ""),
                "content": r.get("snippet", "")
            })

        return json.dumps(output, ensure_ascii=False)

    except Exception as e:
        print(f"[TOOL ERROR]: {e}")
        return json.dumps([{"title": "Error", "source": "", "content": f"Failed to fetch results: {e}"}])


def document_scraper(url: str) -> str:
    """
    Scrapes a given URL to retrieve main document content.
    """
    print(f"[TOOL]: Scraping content from: '{url}'")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract all paragraph text
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        content = "\n".join(paragraphs)

        if not content.strip():
            content = "No readable content found."

        return content
    except Exception as e:
        print(f"[TOOL ERROR]: {e}")
        return f"Failed to scrape content from {url}"


def update_vector_db(content: str, source: str) -> str:
    """Embeds the content, chunks it if necessary, and stores in FAISS + metadata store."""
    print(f"[TOOL]: Embedding and storing document from {source}")

    # Step 1: Chunk the content
    chunks = chunk_text(content, max_tokens=100)

    # Step 2: Encode chunks
    embeddings = EMBEDDING_MODEL.encode(chunks, normalize_embeddings=True)

    # Step 3: Add to FAISS
    FAISS_INDEX.add(np.array(embeddings, dtype="float32"))

    # Step 4: Save metadata
    for chunk in chunks:
        DOC_STORE.append({"content": chunk, "source": source})

    return f"Stored {len(chunks)} chunks from {source} successfully."

# --------------------------------------------------------------------------

# --- AUTOGEN SETUP ---

# 1. Configuration (Replace with your actual configuration)
config_list = [
    {
        "model": "gpt-4-turbo",
        "provider": "openai",
        "api_key": "YOUR_API_KEY",
        "temperature": 0,       # optional
        "max_tokens": 2000,     # optional
    }
]


# 2. Define the Agents and Roles

# The agent that executes the tools and initiates the chat
user_proxy = UserProxyAgent(
    name="Admin"
)

# The agent responsible for finding, scraping, and orchestrating the data flow
search_agent = AssistantAgent(
    name="SearchAgent",
    llm_config={"config_list": config_list},
    system_message="""
    You are a dedicated article searcher responsible for populating the vector database.
    Your task is a multi-step process:
    1. **Search:** Call `web_search(query)` to find relevant article sources. **The output is a JSON string which you MUST parse to find the URLs.**
    2. **Scrape:** For each source URL found in the JSON, call `document_scraper(url)` to get the full article content.
    3. **Ingest:** For each scraped article, instruct the VectorDBManager to call `update_vector_db(content, source)` with the full scraped content and the original URL.
    4. **Terminate:** Once 2-3 articles are processed and stored, reply with TERMINATE.
    """,
    tools=[web_search, document_scraper]
)

# The agent responsible for database ingestion
vector_db_manager = AssistantAgent(
    name="VectorDBManager",
    llm_config={"config_list": config_list},
    system_message="""
    You are a professional database manager. Your only function is to receive content and source 
    from the SearchAgent and call the `update_vector_db` function to store it. 
    You must only use the provided function and report the result back to the SearchAgent.
    """,
    tools=[update_vector_db]
)


'''
# 3. Register the Tools with the Agents
# All tools are executed by the user_proxy

register_function(
    web_search,
    caller=search_agent,
    executor=user_proxy,
    name="web_search"
)

register_function(
    document_scraper,
    caller=search_agent,
    executor=user_proxy,
    name="document_scraper"
)

# update_vector_db is called by the VectorDBManager
register_function(
    update_vector_db,
    caller=vector_db_manager,
    executor=user_proxy,
    name="update_vector_db"
)
'''

# 4. Define the Group Chat and Manager
groupchat = RoundRobinGroupChat(
    participants=[user_proxy, search_agent, vector_db_manager],
    max_round=15
)


# 5. Initiate the Process
topic = "The latest advancements in quantum computing"
print(f"--- Starting AutoGen workflow for topic: {topic} ---")


initial_message = f"I need you to find and store all relevant articles on the topic: '{topic}' in the vector database by searching for sub topics relevant to the topic"


groupchat.run(
    llm_config={"config_list": config_list},
    messages=[{"role": "user", "content": initial_message}]
)


# 6. Verification (Optional)
print("\n--- Final Verification ---")
print(f"Total documents stored in DOC_STORE: {len(DOC_STORE)}")
if DOC_STORE:
    print(f"Example chunk content: {DOC_STORE[0]['content']}")
    print(f"Source: {DOC_STORE[0]['source']}")
print(f"Number of vectors in FAISS index: {FAISS_INDEX.ntotal}")
