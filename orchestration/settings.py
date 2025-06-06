from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Setting(BaseSettings):
    user_artifact_bucket: str = Field(
        description="S3 bucket where to store user artifacts",
    )
    rag_endpoint: str = Field(
        description="RAG server endpoint",
    )
    anns_endpoint: str = Field(
        description="Annotations server endpoint",
    )
    openai_api_key: str = Field(
        description="OpenAI API key",
    )
    msgapi_endpoint: str = Field(
        default="http://msgapi-svc.default.svc.cluster.local",
        description="SageMaker Message API endpoint"
    )
    use_local_slashgpt: bool = Field(
        required=False,
        default=False,
        description="Use local installation of SlashGPT"
    )
    random_seed: int = Field(
        required=False,
        default=12345678,
        description="Random seed"
    )
    max_retries: int = Field(
        required=False,
        default=3,
        description="Max server retries"
    )


settings = Setting()