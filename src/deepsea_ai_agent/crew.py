import os
from crewai import Agent, Crew, Process, Task,LLM
from crewai.project import CrewBase, agent, crew, task

from crewai_tools import SerperDevTool
from deepsea_ai_agent.tools.custom_tool import TavilySearchTool, JinaReaderTool
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model=os.getenv("MODEL"),
    api_key=os.getenv("GROQ_API_KEY")
    )

@CrewBase
class DeepSeaCrew():

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            verbose=True,
             tools=[
                SerperDevTool(),
                TavilySearchTool(),
                JinaReaderTool()
            ]
        )

    @agent
    def strategist(self) -> Agent:
        return Agent(
            config=self.agents_config["strategist"],
            verbose=True,
             tools=[
                SerperDevTool(),
                TavilySearchTool()
              
            ]
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"]
        )

    @task
    def reporting_analyst(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_analyst"]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            llm=llm
        )