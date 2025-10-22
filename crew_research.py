from crewai import Agent, Task, Crew
from textwrap import dedent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from typing import List, Type
from litellm import completion
from rag import EMBEDDING_MODEL, FAISS_INDEX, DOC_STORE
import numpy as np


class RAGInput(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(3, description="Number of top documents to retrieve")

class RetrievalAugmentedGenerationTool(BaseTool):
    name: str = "Retrieval Augmented Generation"
    description: str = """Retrieves top-k relevant chunks from FAISS.
    The input to this tool should be a question or search query.
    It returns a list of dictionaries, where each dictionary represents a relevant document chunk.
    """
    args_schema: Type[BaseModel] = RAGInput

    def _run(self, query: str, k: int = 3) -> List[dict]:
        """Retrieves top-k relevant chunks from FAISS."""
        query_emb = EMBEDDING_MODEL.encode([query], normalize_embeddings=True)
        D, I = FAISS_INDEX.search(np.array(query_emb, dtype="float32"), k)

        results = []
        for idx in I[0]:
            if idx < len(DOC_STORE):
                results.append(DOC_STORE[idx])
        return results


retrieval_tool = RetrievalAugmentedGenerationTool()

llm = ChatOpenAI(
            model_name="", # Use an OpenRouter-compatible model name
            temperature=0,
            api_key="", # Your OpenRouter API key
            base_url="https://openrouter.ai/api/v1" 
        )


class CustomAgents:

    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="", # Use an OpenRouter-compatible model name
            temperature=0,
            api_key="", # Your OpenRouter API key
            base_url="https://openrouter.ai/api/v1" 
        )
    
    def retriever_agent(self):

        return Agent(
            role="Information Retriever",
            backstory=dedent(f"""You know what are the various relevant concepts in any topic. You can fetch the relevant data related to any topic by making appropiate queries to a RAG system"""),
            goal=dedent(f"""Get all the relevant chunks of text related to a given topic from available documents"""),
            tools=[retrieval_tool],
            verbose=True,
            llm=self.llm,
            deferred=True,
        )

    def writer_agent(self):
        return Agent(
            role="Writer",
            backstory=dedent(f"""All your life you have loved writing summaries."""),
            goal=dedent(f"""Take the chunks of text from the retriever agent summarize it nicely."""),
            verbose=True,
            llm=self.llm,
            deferred=True,
        )


class CustomTasks:
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"

    def retrieval_task(self, agent, topic, suggestions):
        return Task(
            description=dedent(
                f"""
            Tell me precisely what I need to know from the RAG tool.
            This is the topic with respect to which you need to look up relevant concepts: {topic} as per the following suggessions: {suggestions}
            
            {self.__tip_section()} 
        """
            ),
            expected_output="All the relevant text chunks",
            agent=agent,
        )

    def writer_task(self, agent, topic, report, suggestions):
        return Task(
            description=dedent(
                f"""
            Take the input from the previous task and the existing report and write a compelling narrative about it centered around the topic '{topic}'
            
            Existing report: {report}

            Suggestions: {suggestions}
                                       
            {self.__tip_section()}
        """
            ),
            expected_output="Give me the title, then brief summary, then bullet points, and a TL;DR.",
            agent=agent,
        )



class CustomCrew:
    def __init__(self, topic, report, suggestions):
        self.topic = topic
        self.report = report
        if suggestions != "":
            self.suggestions = suggestions
        else: 
            self.suggestions = "The report should be comprehensive and well written"


    def run(self):
        agents = CustomAgents()
        tasks = CustomTasks()

        retriever_agent = agents.retriever_agent()
        writer_agent = agents.writer_agent()

        task1 = tasks.retrieval_task(
            retriever_agent,
            self.topic,
            self.suggestions
        )

        task2 = tasks.writer_task(
            writer_agent,
            self.topic,
            self.report,
            self.suggestions

        )

        crew = Crew(
            agents=[retriever_agent, writer_agent],
            tasks=[task1, task2],
            verbose=True,
        )

        result = crew.kickoff()
        return result
