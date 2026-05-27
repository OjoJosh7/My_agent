from google.adk.agents.llm_agent import Agent
import uuid
from google.genai import types
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)  # <---- 1. Import the Plugin
import asyncio

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

print("✅ ADK components imported successfully.")

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",  # Run MCP server via npx
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter=["getTinyImage"],
        ),
        timeout=30,
    )
)


LARGE_REQUEST_THRESHOLD = 1


def place_image_request(
    num_images: int, tool_context: ToolContext
) -> dict:
    """Places an image request. Requires approval if request is more than 1 image (LARGE_ORDER_THRESHOLD).

    Args:
        num_images: Number of images to generate
        

    Returns:
        Dictionary with request status
    """
     # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 1: A single request (1 image) auto-approve
    if num_images <= LARGE_REQUEST_THRESHOLD:
        return {
            "status": "approved",
            "request_id": f"ORD-{num_images}-AUTO",
            "num_images": num_images,
            "message": f"Request auto-approved: {num_images} images",
        }

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 2: This is the first time this tool is called. Large request need human approval -

    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"⚠️ Large request: {num_images} images. Do you want to approve?",
            payload={"num_images": num_images},
        )
        return {  # This is sent to the Agent
            "status": "pending",
            "message": f"request for {num_images} images requires approval",
        }
    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 3: The tool is called AGAIN and is now resuming. Handle approval response - RESUME here.

    if tool_context.tool_confirmation.confirmed:
        return {
            "status": "approved",
            "request_id": f"ORD-{num_images}-HUMAN",
            "num_containers": num_images,
            "message": f"Request approved: {num_images} images",
        }
    else:
        return {
            "status": "rejected",
            "message": f"Request rejected: {num_images} images",
        }
    
# Create image agent with MCP integration
image_agent = LlmAgent(
    name="image_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are an image generator assistant.
  
  When users request to generate an image:
   1. Use the place_image_request tool with the number of images
   2. If the request status is 'pending', inform the user that approval is required
   3. After receiving the final result, provide a clear summary including:
      - Order status (approved/rejected)
      - Order ID (if available)
      - Number of images
   4. Use the MCP Tool to generate images for user queries if the request is approved
   5. Keep responses concise but informative
  """,
    tools=[mcp_image_server, FunctionTool(func=place_image_request)],
)

# Wrap the agent in a resumable app - THIS IS THE KEY FOR LONG-RUNNING OPERATIONS!
# It keeps the state of the agent
image_app = App(
    name="image_coordinator",
    root_agent=image_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

session_service = InMemorySessionService()

# Create runner with the resumable app
image_runner = Runner(
    app=image_app,  # Pass the app instead of the agent
    session_service=session_service,
)

def check_for_approval(events):
    """Check if events contain an approval request.

    Returns:
        dict with approval details or None
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None

def print_agent_response(events):
    """Print agent's text responses from events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")

def create_approval_response(approval_info, approved):
    """Create approval response message."""
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", parts=[types.Part(function_response=confirmation_response)]
    )

async def run_image_workflow(query: str, auto_approve: bool = True):
    """Runs a image request workflow with approval handling.

    Args:
        query: User's image request
        auto_approve: Whether to auto-approve large orders (simulates human decision)
    """

    print(f"\n{'='*60}")
    print(f"User > {query}\n")

    # Generate unique session ID
    session_id = f"order_{uuid.uuid4().hex[:8]}"

    # Create session
    await session_service.create_session(
        app_name="image_coordinator", user_id="test_user", session_id=session_id
    )

    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    events = []

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # STEP 1: Send initial request to the Agent. If num_images > 1, the Agent returns the special `adk_request_confirmation` event
    async for event in image_runner.run_async(
        user_id="test_user", session_id=session_id, new_message=query_content
    ):
        events.append(event)

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # STEP 2: Loop through all the events generated and check if `adk_request_confirmation` is present.
    approval_info = check_for_approval(events)

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # STEP 3: If the event is present, it's a large order - HANDLE APPROVAL WORKFLOW
    if approval_info:
        print(f"⏸️  Pausing for approval...")
        print(f"🤔 Human Decision: {'APPROVE ✅' if auto_approve else 'REJECT ❌'}\n")

        # PATH A: Resume the agent by calling run_async() again with the approval decision
        async for event in image_runner.run_async(
            user_id="test_user",
            session_id=session_id,
            new_message=create_approval_response(
                approval_info, auto_approve
            ),  # Send human decision here
            invocation_id=approval_info[
                "invocation_id"
            ],  # Critical: same invocation_id tells ADK to RESUME
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent > {part.text}")

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    else:
        # PATH B: If the `adk_request_confirmation` is not present - no approval needed - order completed immediately.
        print_agent_response(events)

    print(f"{'='*60}\n")



