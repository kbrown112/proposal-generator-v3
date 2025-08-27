from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai import LLM
import os
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

def get_azure_llm():
    """Create Azure OpenAI LLM with proper authentication"""
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "api://ailab/Model.Access"
    )
    
    return LLM(
        model=os.environ.get("MODEL"),
        api_base=os.environ.get("API_BASE"),
        api_version=os.environ.get("API_VERSION"),
        azure_ad_token_provider=token_provider
    )


@CrewBase
class ProposalGenerator():
    """ProposalGenerator crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def proposal_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['proposal_expert'], # type: ignore[index]
            verbose=True,
            llm=get_azure_llm()
        )

    @agent
    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config['manager'], # type: ignore[index]
            verbose=True,
            llm=get_azure_llm()
        )
    
    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst'], # type: ignore[index]
            verbose=True,
            llm=get_azure_llm()
        )
    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def proposal_expert_task(self) -> Task:
        return Task(
            config=self.tasks_config['proposal_expert_task'], # type: ignore[index]
        )

    @task
    def manager_task(self) -> Task:
        return Task(
            config=self.tasks_config['manager_task'], # type: ignore[index]
        )
    
    @task
    def analyst_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyst_task'], # type: ignore[index]
            output_file='solution_summary.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ProposalGenerator crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
