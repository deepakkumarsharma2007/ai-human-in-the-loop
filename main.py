from dotenv import load_dotenv
from fastapi import FastAPI
from api.router import conversations_api_router
from api.health_check_routes import health_check_router
from api.health_check_routes import mark_startup_complete
from genai_core.logs.agent_logging import DKSAgentLogger

load_dotenv()


# To embed a dicuemnt run the below
# def main():
#     load_dotenv()
#     from genai_core.rag.script_to_embed import embed
#     embed() #embedding is done




async def lifespan(app: FastAPI):
    try:
        mark_startup_complete()
        DKSAgentLogger.get_logger().info("Application startup completed successfully")
    except Exception as e:
        DKSAgentLogger.get_logger().error(f"Error during startup: {str(e)}")
        # Mark as complete anyway to allow the app to start
        mark_startup_complete()
    yield
    

app = FastAPI(
    # root_path=ROOT_PATH,
        title="AI Orchestrator API",
        docs_url="/docs",
        openapi_url="/openapi.json",
        redoc_url=None,
        lifespan=lifespan)


# Add health check router
app.include_router(health_check_router)

api_prefix = "/api/chat"

app.include_router(conversations_api_router, prefix=api_prefix)