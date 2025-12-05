from coordinator_agent.agent import root_agent

from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
import asyncio

APP_NAME = "coordinator_agent"
USER_ID = "user_123"
SESSION_ID = "session_123"


# --- Setup Session + Runner ---
async def setup_session_and_runner():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    return session_service, session, runner


# --- Run the Agent and Track State ---
async def call_agent_async(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    session_service, session, runner = await setup_session_and_runner()

    print(f"🟢 Initial session state: {session.state}")

    events = runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content
    )

    async for event in events:
        if event.is_final_response():
            if event.content and event.content.parts:
                text_part = next((p.text for p in event.content.parts if p.text), None)
                print("🗣️ Agent Response:", text_part or "[No textual response]")
            else:
                print("🗣️ Agent Response: [Empty or non-text response]")

    # --- Check Updated State ---
    updated_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print(f"🟣 Updated session state: {updated_session.state}")


# --- Run Example ---
if __name__ == "__main__":
    asyncio.run(call_agent_async("Deep Neural Network"))