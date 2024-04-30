import os

import dotenv


def set_openai_api_key(openai_env_file):
    dotenv.load_dotenv(openai_env_file)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key
