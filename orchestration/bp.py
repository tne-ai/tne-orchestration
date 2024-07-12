import os
import json
import boto3
import base64
import pandas as pd
from io import StringIO
from typing import Dict, List, Optional, Union

from orchestration.db_utils import PostgresConnector, RajkumarFormatter

# Uncomment below to use local SlashGPT
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "../../SlashTNE/src"))

from slashgpt.chat_config_with_manifests import ChatConfigWithManifests

BUCKET_NAME = "bp-authoring-files"


class MalformedDataError(Exception):
    def __init__(self, message):
        super().__init__(message)


class BPServerConfig:
    def __init__(
        self,
        agent_name: Optional[str] = "TNE",
        timeout: Optional[int] = 120,
        max_retries: Optional[int] = 3,
        max_rows: Optional[int] = 10000,
    ):
        # Max number of seconds to wait for an API call to return
        self.timeout = timeout
        # Max number of data rows to display
        self.max_rows = max_rows
        # Agent name
        self.agent_name = agent_name
        # Max number of LLM API call retries before raising an error
        self.max_retries = max_retries


class BP:
    def __init__(
        self,
        process: dict,
        uid: str,
        cache_file: str = "./schema_cache.json",
    ):
        # Server configs for the process
        self.uid = uid
        self.server_config = BPServerConfig()

        # Load the process file components
        try:
            self.name = process["name"]
            self.data = process["data"]
            self.steps_list = process["steps"]
            self.descriptions = process["description"]
            self.manifests_path = "."
        except KeyError as k:
            raise MalformedDataError(f"Process had parsing error: {k}")

        # LLM configs
        manifest_s3_path = f"d/{self.uid}/manifests"
        self.slashgpt_config = ChatConfigWithManifests(
            base_path="", path_manifests=self.manifests_path
        )
        self.manifests = self.slashgpt_config.load_manifests_s3(
            BUCKET_NAME, manifest_s3_path
        )

        # Load data sources
        db_metadata = self.__load_data(cache_file)
        self.db_connectors = db_metadata["connectors"]

        # Keep track of state
        self.current_step = 0

        # Load steps
        self.steps = self.__load_steps(process["steps"])

    def __load_data(self, cache_file: str) -> Optional[Dict]:
        """Loads the manifests from files into memory and uses them to pull the latest schemas."""
        # Get schemas for database in each manifest
        db_metadata = {
            "schemas": {},
            "connectors": {},
            "mappings": {},
        }
        for k in self.data.keys():
            source = self.data.get(k)
            if source.get("type") == "PostgreSQL":
                # Cache database schemas as to not repeatedly pull
                if os.path.exists(cache_file):  # TODO: need cache timer
                    with open(cache_file, "r") as fp:
                        schema_cache = json.load(fp)
                else:
                    schema_cache = {}
                if k not in schema_cache.keys():
                    try:
                        db_creds = {}
                        cred_envvars = source.get("creds")
                        for field in cred_envvars.keys():
                            db_creds[field] = os.getenv(cred_envvars[field])
                        db_connector = self.__connect_db(k, **db_creds)
                        prompt_formatter = self.__get_formatter(db_connector)
                        schema = prompt_formatter.pull_schema(aliased=False)
                        # Write to cache
                        schema_cache.update({k: schema})
                        with open(cache_file, "a") as fp:
                            json.dump(schema_cache, fp)
                    except Exception as e:
                        raise ValueError(
                            f"Got unexpected error while trying to pull schema for database [{k}]: {e}"
                        )
                else:
                    db_creds = {}
                    schema = schema_cache.get(k)
                    cred_envvars = source.get("creds")
                    for field in cred_envvars.keys():
                        db_creds[field] = os.getenv(cred_envvars[field])
                    db_connector = self.__connect_db(k, **db_creds)
                    # TODO: need to implement caching support for these
                    prompt_formatter = None

                self.data.get(k)["data"] = schema
                db_metadata["connectors"][k] = db_connector
                if prompt_formatter:
                    db_metadata["mappings"][k] = prompt_formatter.mappings

        return db_metadata

    def __load_steps(self, steps: List):
        steps_list = []
        for step in steps:
            if type(step) is dict:
                steps_list.append(ProcessStep(self, step, self.uid))
            elif type(step) is list:
                parallel_step = []
                for s in step:
                    parallel_step.append(ProcessStep(self, s, self.uid))
                steps_list.append(parallel_step)

        return steps_list

    @classmethod
    def __connect_db(
        cls,
        dbname: str = "ebp",
        user: str = "",
        password: str = "",
        host: str = "",
        port: int = 5432,
    ):
        db_connector = PostgresConnector(
            user=user, password=password, dbname=dbname, host=host, port=port
        )

        db_connector.connect()
        return db_connector

    @classmethod
    def __get_formatter(cls, db_connector: PostgresConnector):
        db_schema, db_data, db_unique_vals = {}, {}, {}
        for table in db_connector.get_tables():
            db_schema[table] = db_connector.get_schema(table)
            db_data[table] = db_connector.select_random(table, num_selections=3)
            db_unique_vals[table] = db_connector.get_distinct_values(table)

        formatter = RajkumarFormatter(
            db_schema, db_data, db_unique_vals, use_unique_vals=True
        )

        return formatter

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_step < len(self.steps):
            current_step = self.steps[self.current_step]
            self.current_step += 1
            return current_step
        else:
            self.current_step = 0
            raise StopIteration


class ProcessStep:
    def __init__(self, parent: BP, proc_step: Dict, uid: str):
        self.parent = parent

        # Required fields
        self.name = proc_step.get("name")
        self.type = proc_step.get("type")
        self.description = proc_step.get("description")

        # Conditional fields
        self.data = {}
        self.data_sources = proc_step.get("data_sources")
        self.output_type = proc_step.get("output_type")
        self.possible_outputs = proc_step.get("possible_outputs")
        self.module = proc_step.get("module")
        self.kwargs = proc_step.get("kwargs")
        self.input = proc_step.get("input")
        self.data_output_name = proc_step.get("data_output_name")
        self.debug_output_name = proc_step.get("debug_output_name")
        self.use_prev_input = proc_step.get("use_prev_input")
        self.sql_queries = proc_step.get("sql_queries")
        self.tool_json = proc_step.get("tool_json")
        self.tool_code = proc_step.get("tool_code")
        self.rag_db_name = proc_step.get("rag_db_name")
        self.suppress_output = proc_step.get("suppress_output")
        self.show_debug = proc_step.get("show_debug")
        self.max_count = proc_step.get("max_count")
        self.min_similarity = proc_step.get("min_similarity")
        self.prompt = proc_step.get("prompt")

        # Function
        self.manifest = None
        if self.type == "llm" or self.type == "llm-python":
            self.manifest = self.parent.manifests.get(self.name.split(".")[0])

        self.module = None
        if self.type == "python":
            self.module = proc_step.get("module")
        if self.type == "python_code":
            self.module = proc_step.get("module")

        # Data config
        if self.data_sources and self.data_sources[0] is not None:
            for i, source in enumerate(self.data_sources):
                if type(source) is list:
                    source = source[0].get("dataSource").strip()
                    data_source = self.parent.data.get(source)
                    if data_source:
                        self.data.update({source: data_source})

                data_source = self.parent.data.get(source)
                if data_source:
                    self.data.update({source: data_source})
