"""
research_agent.py

An autonomous, multi-agent system for generating a research report.

"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import json
from typing import TypedDict, List, Dict
from collections import deque
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from crew_research import CustomCrew, llm   # import the class
from rag import EMBEDDING_MODEL, FAISS_INDEX, DOC_STORE
from langchain_community.tools import tool as langchain_tool
from langgraph.graph import StateGraph, END



# --- Configuration & State Management ---
class AgentState(TypedDict):
    """Represents the state of the agent workflow."""
    goal: str
    subtasks: List[str]
    current_task: str
    research_results: List[Dict]
    draft_summary: str
    final_report: str
    suggestions: str
    reasoning_trace: List[Dict]
    citations: List[str]
    vector_db: Dict
    message_queue: deque
    retries: int
    max_retries: int
    num_verify: int

IN_MEMORY_VECTOR_DB = {}
TASK_QUEUE = deque()

tavily_client = TavilyClient(api_key="") #Your Tavily API key


LLM_CACHE_FILE = "llm_cache.json"
SEARCH_CACHE_FILE = "tavily_cache.json"

# Load caches if they exist
def load_cache(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

llm_cache = load_cache(LLM_CACHE_FILE)
search_cache = load_cache(SEARCH_CACHE_FILE)


def cached_llm(goal):
    """
    Return cached LLM subtasks for the goal if available.
    Otherwise, invoke the LLM and save the result.
    """
    if goal in llm_cache:
        print("Using cached LLM subtasks for goal:", goal)
        return llm_cache[goal]

    # Call your LLM
    response = llm.invoke(
        f"Break down the goal '{goal}' into a list of 3 concise research subtasks. "
        "Respond with only a JSON list of strings, e.g., ['Task 1', 'Task 2']."
    )
    
    clean_output = re.sub(r"^```(?:json)?|```$", "", response.content.strip(), flags=re.MULTILINE).strip()
    subtasks = json.loads(clean_output)

    # Save to cache
    llm_cache[goal] = subtasks
    with open(LLM_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(llm_cache, f, ensure_ascii=False, indent=2)

    print("Saved LLM subtasks to cache for goal:", goal)
    return subtasks



def cached_llm_verifier(goal, report):
    
    # TODO Implement cache
    # Call your LLM
    response = llm.invoke(
    f"""Check the following main body of report on the topic '{goal}' and provide suggestions for improving the body.
    Respond with a JSON object with a single key "suggestions" whose value is a single string containing the suggestions.
    If no major changes are required, respond with: {{"suggestions": ""}}.

    Report body:
    \"\"\"{report}\"\"\"
    """
    )

    response_content = response.content.strip()
    # Step 1: Remove ```json or ``` code fences
    cleaned = re.sub(r"^```(?:json)?|```$", "", response_content, flags=re.MULTILINE).strip()

    # Step 2: Extract the suggestions string (handles single/double quotes, optional spaces)
    match = re.search(
        r"""["']?suggestions["']?\s*:\s*["']([^"']*)["']""",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL
    )

    suggestions = match.group(1) if match else ""
    
    return suggestions



def cached_search(query, num_results=5):
    """
    Check if the query result exists in cache. 
    If not, perform the Tavily search and store the result.
    """
    if query in search_cache:
        print("Using cached result for query:", query)
        return search_cache[query]
    
    # Perform the actual Tavily search
    results = tavily_client.search(query, num_results=num_results)
    
    # Save to cache
    search_cache[query] = results
    with open(SEARCH_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(search_cache, f, ensure_ascii=False, indent=2)
    
    print("Saved result to cache for query:", query)
    return results


def chunk_text(text: str, max_tokens: int = 100) -> List[str]:
    """Simple text chunker by words."""
    words = text.split()
    return [
        " ".join(words[i:i+max_tokens])
        for i in range(0, len(words), max_tokens)
    ]


# --- Agentic Tools ---


@langchain_tool
def web_search(query: str) -> str:
    """
    Performs a web search using Tavily API to find relevant information.

    Args:
        query (str): The search query.

    Returns:
        str: JSON-formatted search results.
    """
    print(f"\n[TOOL]: Performing web search for: '{query}'")

    try:
        # Call Tavily API with a query
        results = cached_search(query, num_results=5)  

        # Convert results to a standardized JSON format
        output = []
        for r in results.get("results", []):
            output.append({
                "title": r.get("title", ""),
                "source": r.get("url", ""),
                "content": r.get("snippet", "")
            })

        return json.dumps(output, ensure_ascii=False)

    except Exception as e:
        # Fallback in case of error
        print(f"[TOOL ERROR]: {e}")
        return json.dumps([{
            "title": "Error",
            "source": "",
            "content": f"Failed to fetch results for '{query}'."
        }])



@langchain_tool
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


@langchain_tool
def update_vector_db(content: str, source: str) -> str:
    """Embeds the content, chunks it if necessary, and stores in FAISS + metadata store."""
    print(f"[TOOL]: Embedding and storing document from {source}")

    chunks = chunk_text(content, max_tokens=100)

    embeddings = EMBEDDING_MODEL.encode(chunks, normalize_embeddings=True)

    FAISS_INDEX.add(np.array(embeddings, dtype="float32"))

    for chunk in chunks:
        DOC_STORE.append({"content": chunk, "source": source})

    return f"Stored {len(chunks)} chunks from {source} successfully."



@langchain_tool
def retrieval_augmented_generation(query: str, k: int = 3) -> List[dict]:
    """Retrieves top-k relevant chunks from FAISS."""
    query_emb = EMBEDDING_MODEL.encode([query], normalize_embeddings=True)
    D, I = FAISS_INDEX.search(np.array(query_emb, dtype="float32"), k)

    results = []
    for idx in I[0]:
        if idx < len(DOC_STORE):
            results.append(DOC_STORE[idx])
    return results

# --- Agent Definitions ---


def researcher_agent(state: AgentState):
    """
    The researcher agent's role is to break down the goal into subtasks,
    use tools to gather information, and store it in the vector database.
    It handles tool-use and error handling.
    """
    print("\n[AGENT]: Researcher is active...")
    print(f"Goal: {state['goal']}")

    if not state.get("subtasks"):
        print("[RESEARCHER]: Breaking down the goal into research subtasks.")
        subtasks = cached_llm(state['goal'])
        state['subtasks'] = subtasks
        print("Sub tasks are ", subtasks)
        state['reasoning_trace'].append({"agent": "Researcher", "action": "Subtask decomposition", "details": subtasks})

    if state['subtasks']:
        state['current_task'] = state['subtasks'].pop(0)
        print(f"[RESEARCHER]: Starting new subtask: '{state['current_task']}'")
        
        try:
            tool_call = web_search.invoke({"query": state['current_task']})
            search_results = json.loads(tool_call)
            
            for result in search_results:
                scraped_content = document_scraper.invoke({"url": result['source']})
                update_vector_db.invoke({"content": scraped_content, "source": result['source']})
            
            state['research_results'].extend(search_results)
            state['reasoning_trace'].append({"agent": "Researcher", "action": "Research and document storage", "details": f"Completed research for task: {state['current_task']}"})
        
        except Exception as e:
            print(f"[RESEARCHER ERROR]: Task failed: {e}. Retrying...")
            state['retries'] += 1
            if state['retries'] > state['max_retries']:
                print("[RESEARCHER]: Max retries reached. Skipping task.")
                state['reasoning_trace'].append({"agent": "Researcher", "action": "Task failure", "details": f"Task '{state['current_task']}' failed after max retries."})
                state['retries'] = 0
            else:
                state['subtasks'].append(state['current_task']) # Add back to queue
                
    return state

def summarizer_agent(state: AgentState):
    """
    The summarizer agent retrieves information from the vector DB and
    synthesizes a draft summary of the research.
    """
    print("\n[AGENT]: Summarizer is active...")
    if not state['research_results']:
        print("[SUMMARIZER]: No research results to summarize. Passing...")
        return state
    
    print("[SUMMARIZER]: Synthesizing a draft report from research results.")
    
    custom_crew = CustomCrew(state['goal'], state['draft_summary'], state['suggestions'])
    result = custom_crew.run()


    state['draft_summary'] = result.raw
    state['reasoning_trace'].append({"agent": "Summarizer", "action": "Drafting report", "details": "Drafted summary from retrieved documents."})
    return state

def verifier_agent(state: AgentState):
    """
    The verifier agent reviews the draft report, adds citations, and finalizes the output.
    It can also request new research if the draft is incomplete.
    """
    print("\n[AGENT]: Verifier is active...")
    print("[VERIFIER]: Reviewing the draft report for accuracy and completeness.")


    if state['num_verify'] >= 3:
        state['suggestions'] = ""
    else:
        suggestions = cached_llm_verifier(state['goal'], state['draft_summary'])
        state['suggestions'] = suggestions
        state['num_verify'] += 1

    if state['suggestions'] != "":
        return state

    
    citations = [f"[{i+1}] {doc['source']}" for i, doc in enumerate(state['research_results'])]
    
        
    final_report = f"Final Market Research Report\n\n**Goal:** {state['goal']}\n\n"
    final_report += state['draft_summary']
    final_report += "\n\n---\n\n**Citations:**\n" + "\n".join(citations)

    state['final_report'] = final_report
    state['citations'] = citations
    state['reasoning_trace'].append({"agent": "Verifier", "action": "Finalizing report", "details": "Verified draft, added citations, and created final report."})
    return state

# --- Workflow Graph Definition ---

def should_continue(state: AgentState):
    """Determines the next step in the workflow."""
    if not state['subtasks']:
        print("[WORKFLOW]: All subtasks completed. Moving to summarization.")
        return "summarize"
    else:
        print(f"[WORKFLOW]: {len(state['subtasks'])} subtasks remaining. Continuing research.")
        return "research"

def should_finalize(state: AgentState):
    """Determines if the report is ready to be finalized."""
    if state['suggestions'] != "":
        print("[WORKFLOW]: Report is not finalized. Passing back to summarizer.")
        return "summarize"
    else:
        print("[WORKFLOW]: Report is finalized. Ending workflow.")
        return "end"



# Initialize the graph
workflow = StateGraph(AgentState)

# Add nodes for each agent
workflow.add_node("research", researcher_agent)
workflow.add_node("summarize", summarizer_agent)
workflow.add_node("verify", verifier_agent)

# Define the entry and exit points
workflow.set_entry_point("research")

# Define the edges (transitions between nodes)
workflow.add_conditional_edges(
    "research",
    should_continue,
    {
        "research": "research",
        "summarize": "summarize",
    }
)
workflow.add_edge("summarize", "verify")
workflow.add_conditional_edges(
    "verify",
    should_finalize,
    {
        "summarize": "summarize",
        "end": END,
    }
)

# Compile the graph
app = workflow.compile()

# --- Main Execution ---
if __name__ == "__main__":
    goal = "Write a report on the Solar Panel industry in Europe."
    
    initial_state = {
        "goal": goal,
        "subtasks": [],
        "current_task": "",
        "research_results": [],
        "draft_summary": "",
        "final_report": "",
        "suggestions": "",
        "reasoning_trace": [],
        "citations": [],
        "vector_db": IN_MEMORY_VECTOR_DB,
        "message_queue": TASK_QUEUE,
        "retries": 0,
        "num_verify": 0,
        "max_retries": 2
    }
    
    print("--- Starting Autonomous Research Workflow ---")
    
    # Run the workflow
    final_state = app.invoke(initial_state)

    print("\n--- Workflow Completed ---")
    print(final_state['final_report'])
    print("\n--- Reasoning Trace ---")
    for step in final_state['reasoning_trace']:
        print(f"- Agent: {step['agent']}\n  Action: {step['action']}\n  Details: {step['details']}")
    print("\n--- End of Report ---")


