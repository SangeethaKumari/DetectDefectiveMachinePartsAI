from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search
from google.adk.runners import InMemoryRunner
import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override=True)


# https://www.kaggle.com/code/kaggle5daysofai/day-1a-from-prompt-to-action?scriptVersionId=275610687&cellId=16
# following this link to create a agent that can answer questions about the data
# Define an async main function
async def main():
    root_agent = Agent(
        name="helpful_assistant",
        model="gemini-2.5-flash-lite",
        description="A simple agent that can answer general questions.",
        instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
        tools=[google_search],
    )

    print("✅ Root Agent defined.")

    runner = InMemoryRunner(agent=root_agent)
    response = await runner.run_debug(
        "What is Agent Development Kit from Google? What languages is the SDK available in?"
    )
    print("Response:", response)



# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())