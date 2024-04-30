from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Setting(BaseSettings):
    user_artifact_bucket: str = Field(
        default="bp-authoring-files",
        description="S3 bucket where to store user artifacts",
    )
    rag_endpoint: str = Field(
        description="RAG server endpoint",
    )
    openai_api_key: str = Field(
        description="OpenAI API key",
    )


settings = Setting()
