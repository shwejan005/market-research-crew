from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class MarketResearchCrew():
    """MarketResearchCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Agents
    @agent
    def market_research_specialist(self) -> Agent:
        return Agent(self.agents_config["market_research_specialist"])

    @agent
    def competitive_intelligence_analyst(self) -> Agent:
        return Agent(self.agents_config["competitive_intelligence_analyst"])
    
    @agent
    def customer_insights_researcher(self) -> Agent:
        return Agent(self.agents_config["customer_insights_researcher"])

    @agent
    def product_strategy_advisor(self) -> Agent:
        return Agent(self.agents_config["product_strategy_advisor"])

    @agent
    def business_analyst(self) -> Agent:
        return Agent(self.agents_config["business_analyst"])
    

    # Tasks
    @task
    def market_research_task(self) -> Task:
        return Task(
            self.tasks_config["market_research_task"]
        )

    @task
    def competitive_intelligence_task(self) -> Task:
        return Task(
            self.tasks_config["competitive_intelligence_task"],
            context= [self.market_research_task()]
        )

    @task
    def customer_insights_task(self) -> Task:
        return Task(
            self.tasks_config["customer_insights_task"],
            context=[self.market_research_task(), self.competitive_intelligence_task()]
        )

    @task
    def product_strategy_task(self) -> Task:
        return Task(
            self.tasks_config["product_strategy_task"],
            context=[self.market_research_task(), self.competitive_intelligence_task(), self.customer_insights_task()]
        )

    @task
    def business_analyst_task(self) -> Task:
        return Task(
            self.tasks_config["business_analyst_task"],
            context=[self.market_research_task(), self.competitive_intelligence_task(), self.customer_insights_task(), self.product_strategy_task()]
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.SEQUENTIAL
        )