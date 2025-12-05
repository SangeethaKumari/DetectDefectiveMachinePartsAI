from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search
from google.adk.runners import InMemoryRunner
import asyncio
import os
from dotenv import load_dotenv
from google.adk.agents.sequential_agent import SequentialAgent
load_dotenv(override=True)


detect_defective_machine_parts_agent = SequentialAgent(
    name="DetectDefectiveMachineParts",
    sub_agents=[
        image_preprocessing_agent,
        image_classification_agent,
    ],
    description="Complete newsletter generation pipeline: Collect → Validate → Analyze → Write → Edit -> Format",
)

# Main entry point for the newsletter generation system
root_agent = detect_defective_machine_parts_agent