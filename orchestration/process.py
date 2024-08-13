import asyncio
import base64
import copy
import inspect

# Temporary function call workarounds to be incorporated into SlashGPT
import json
import logging
import tempfile
import os
import platform
import re
import subprocess
import sys
import time
import runpy
from io import StringIO
from typing import Any, Dict, Union, Tuple, Optional, AsyncGenerator

import boto3
import pandas as pd
import requests
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from mypy_boto3_sagemaker_runtime.client import SageMakerRuntimeClient
from openai import AsyncOpenAI
from opentelemetry import trace
from orchestration.bp import BP, ProcessStep
from orchestration.server_utils import (
    generate_stream,
    get_s3_proc,
    get_s3_ls,
    get_s3_dir_summary,
    get_python_s3_module,
    fetch_python_module,
    get_data_from_s3,
    upload_to_s3,
    parse_s3_proc_data_no_regex,
    is_base64_image,
    create_rag_request,
    create_anns_request,
)
from orchestration.settings import settings
from orchestration.v2.api.api import RagResponse
from orchestration.v2.api.util import (
    async_iterate_streaming_request_generator,
    anns_request_to_json_str,
    anns_response_from_json_str,
)
from pydantic import BaseModel
from slashgpt.chat_session import ChatSession
from tabulate import tabulate
from tne.TNE import TNE

import krt

# Uncomment below to use local SlashGPT
# sys.path.append(os.path.join(os.path.dirname(__file__), "../../SlashTNE/src"))
# from slashgpt.chat_session import ChatSession

logger = logging.getLogger(__name__)

# Literal constants
TNE_PACKAGE_PATH = "./tne-0.0.1-py3-none-any.whl"
image_models = ["dall-e-3"]

DATA_BUFFER_LENGTH = 2500

smr_client = boto3.client("sagemaker-runtime")  # type: SageMakerRuntimeClient

if platform.system() == "Darwin":
    # So that input can handle Kanji & delete
    import readline  # noqa: F401


class _Base(BaseModel):
    pass


class GraphParseError(Exception):
    def __init__(self, message):
        super().__init__(message)


class FlowLog(BaseModel):
    message: Optional[str] = None
    """Log message"""
    error: Optional[str] = None
    """Error message"""


class LLMResponse(BaseModel):
    text: Optional[str] = None
    """Text outputted by the LLM"""
    data: Optional[Union[str, pd.DataFrame]] = None
    """Data outputted by the LLM"""

    class Config:
        arbitrary_types_allowed = True


def replace_escaped_newlines(chunk: str) -> str:
    return chunk.replace("\\n", "\n")


def update_data_context_buffer(session, file_name, data_context_buffer):
    try:
        data = session.get_object(file_name)
        match type(data):
            case pd.DataFrame:
                data_context_buffer += f"{file_name}: {data.head().to_string()}\n"
            case str:
                if len(data) <= DATA_BUFFER_LENGTH:
                    data_context_buffer += data
                else:
                    data_context_buffer += data[:DATA_BUFFER_LENGTH]
    except ValueError as ve:
        data = session.get_object_bytes(file_name).decode("utf-8")
        if len(data) <= DATA_BUFFER_LENGTH:
            data_context_buffer += data
        else:
            data_context_buffer += data[:DATA_BUFFER_LENGTH]
    except IOError as ie:
        return data_context_buffer

    return data_context_buffer


def execute_temp_file(file_path: str) -> dict:
    return runpy.run_path(file_path)


def save_code_to_temp_file(code: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    with open(temp_file.name, "w") as f:
        f.write(code)
    return temp_file.name


async def collect_messages(generator: AsyncGenerator):
    """Collect all messages from an async generator into a list."""
    collected_messages = []
    async for message in generator:
        if type(message) is not FlowLog:
            collected_messages.append(message)

    response_str = ""
    for msg in collected_messages:
        if type(msg) is tuple:
            response_str += msg[0]
        if type(msg) is str:
            response_str += msg
    return response_str


async def vega_chart(
    df_data: pd.DataFrame, uid: str, max_retries: int = 3
) -> Optional[str]:
    retry_no = 0
    chart = None
    plot_chart_proc = get_s3_proc("Plot Chart", "SYSTEM", settings.user_artifact_bucket)
    proc_step = plot_chart_proc.steps[0]
    while retry_no < max_retries and not chart:
        tools = [json.loads(proc_step.tool_json)]
        client = AsyncOpenAI(
            api_key=os.getenv(
                "OPENAI_API_KEY",
            ),
        )
        prompt = proc_step.manifest.get("prompt")
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Plot the data: {df_data.head(5)}",
                },
            ],
            tools=tools,
        )

        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            function_call = choice.message.tool_calls[0].function
            func_args = json.loads(choice.message.tool_calls[0].function.arguments)
            try:
                func_namespace = {}
                func_name, func_code = get_python_s3_module(
                    function_call.name, "SYSTEM", settings.user_artifact_bucket
                )
                exec(func_code, func_namespace)

                func = func_namespace[function_call.name]
                chart = await func(data=df_data, **func_args)
                if chart:
                    return chart.to_json()
                else:
                    retry_no += 1

            except Exception as e:
                return None

    return None


class BPAgent:
    def __init__(self, callback=None):
        self._callback = callback or self._noop
        self.orig_question = ""

    def _noop(self, callback_type, data):
        pass

    def _process_event(self, callback_type, data):
        self._callback(callback_type, data)

    # __run_step is a wrapper for __run_step_inner_impl.
    # Its purpose is to setup a new enclosing OTel span.
    # Only __run_step should call __run_step_inner_impl.
    async def __run_step(
        self,
        proc_step,
        proc,
        step_input,
        dispatched_input,
        uid,
        session_id,
        is_spinning,
        show_description=True,
    ) -> AsyncGenerator:
        tracer = trace.get_tracer(__name__)
        fun_name = "BPAgent.run_step"
        span_name = f'{fun_name}("{proc_step.type}", "{proc_step.name}")'
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("tne.orchestration.fun_name", fun_name)
            span.set_attribute("tne.orchestration.step_type", proc_step.type)
            span.set_attribute("tne.orchestration.step_name", proc_step.name)
            async for message in self.__run_step_inner_impl(
                proc_step,
                proc,
                step_input,
                dispatched_input,
                uid,
                session_id,
                is_spinning,
                show_description=show_description,
            ):
                yield message

    # Only __run_step should call __run_step_inner_impl.
    async def __run_step_inner_impl(
        self,
        proc_step,
        proc,
        step_input,
        dispatched_input,
        uid,
        session_id,
        is_spinning,
        show_description=True,
    ) -> AsyncGenerator:
        # TEMPORARY: route all tne-branded models to groq
        if proc_step.manifest:
            manifest_model = proc_step.manifest.get("model")
            if manifest_model:
                model_name = manifest_model.get("model_name")

        # Handle special case where a LLM step picks from a list of manifests to run
        if proc_step.name == "dispatched":
            proc_step.manifest = proc.manifests.get(step_input)
            step_input = dispatched_input
        if proc_step.input:
            if type(step_input) is str and not is_base64_image(step_input):
                step_input = f"{proc_step.input}\n\n{step_input}"
            else:
                step_input = proc_step.input

        if show_description:
            yield f"**{proc_step.description}**", True
            yield "\n\n", True

        step_output = LLMResponse()

        # Run LLM-based step
        if proc_step.type == "llm":
            llm_step_messages = []

            # Manifest doesn't exist; something is wrong
            if not proc_step.manifest:
                raise AttributeError(f"Missing model file for {proc_step.name}")

            # Function call
            elif not proc_step.manifest.get("model") and proc_step.manifest.get(
                "functions"
            ):
                proc_step.manifest.update({"stream": False})
                collected_messages = []
                try:
                    async for message in self.__run_llm_step(
                        step_input, proc_step, uid
                    ):
                        if type(message) is not FlowLog:
                            collected_messages.append(message)
                        yield message
                except Exception as e:
                    raise e
                llm_step_output = "".join(collected_messages)
                regex_pattern = "```"
                yield FlowLog(
                    message=f"[Assistant][call_llm] Extracting data from LLM response with pattern {regex_pattern}"
                )
                formatted_output = self.__parse_llm_response(
                    llm_step_output, regex_pattern
                )
                if type(formatted_output) is FlowLog:
                    if formatted_output.message:
                        step_output.text = llm_step_output
                    yield formatted_output
                elif type(formatted_output) is str or type(formatted_output) is list:
                    step_output.text = formatted_output

            # Determine if this is vision or text model
            elif proc_step.manifest.get("model").get("model_name") in image_models:
                try:
                    async for message in self.__run_llm_step(
                        step_input, proc_step, uid
                    ):
                        if type(message) is FlowLog:
                            yield message
                        else:
                            llm_step_messages.append(message)

                    img_url = "".join(llm_step_messages)

                    # Generate a unique filename based off of the image contents
                    data_s3_path = f"d/{uid}/data"
                    data_filenames = get_s3_ls(settings.user_artifact_bucket, data_s3_path)

                    assistant_proc = get_s3_proc("Assistant", "SYSTEM", settings.user_artifact_bucket)
                    filename_gen_manifest = assistant_proc.manifests.get(
                        "imageFilename"
                    )
                    filename_gen_manifest["prompt"] = (
                        filename_gen_manifest["prompt"] + f"\n\n{data_filenames}"
                    )

                    collected_messages = []
                    try:
                        async for message in self.process_llm(
                            img_url,
                            None,
                            uid,
                            filename_gen_manifest,
                            session_id=session_id,
                        ):
                            if type(message) is not FlowLog:
                                collected_messages.append(message)
                        img_filename = "".join(collected_messages)
                    except Exception as e:
                        raise e

                    response = requests.get(img_url)
                    if response.status_code == 200:
                        img_s3_url = await upload_to_s3(
                            img_filename, response.content, uid, settings.user_artifact_bucket
                        )
                        step_output.text = f"![]({img_s3_url})"
                        step_output.data = base64.b64encode(response.content).decode(
                            "utf-8"
                        )
                        yield FlowLog(
                            message=f"[BPAgent][run_proc] Uploaded {img_filename} to S3..."
                        )
                    else:
                        yield FlowLog(
                            error=f"[BPAgent][run_proc] Failed to generate image..."
                        )
                        raise IOError("Failed to generate image")
                except Exception as e:
                    raise e
            else:
                try:
                    async for message in self.__run_llm_step(
                        step_input, proc_step, uid
                    ):
                        if type(message) is not FlowLog:
                            llm_step_messages.append(message)
                        yield message
                    yield "\n"
                except Exception as e:
                    raise e
                llm_step_output = "".join(llm_step_messages)
                # Look for special outputs enclosed within backticks
                regex_pattern = "```"
                yield FlowLog(
                    message=f"[Assistant][call_llm] Extracting data from LLM response with pattern {regex_pattern}"
                )
                formatted_output = self.__parse_llm_response(
                    llm_step_output, regex_pattern
                )
                if type(formatted_output) is FlowLog:
                    if formatted_output.message:
                        step_output.text = llm_step_output
                    yield formatted_output
                elif type(formatted_output) is str or type(formatted_output) is list:
                    step_output.text = formatted_output

        # Run Python code and may return text or data - DEPRECATED
        elif proc_step.type == "python":
            python_step_resp = None
            try:
                python_step_resp = await self.__run_python_step(
                    step_input, proc_step, uid
                )
            except ValueError as e:
                err_str = e.__str__()
                step_output.text = f"Received the following error during code execution. Please adjust accordingly and try again.\n\n{err_str}"
                yield step_output.text
                yield "\n"
            if type(python_step_resp) is str:
                step_output.text = python_step_resp
                yield python_step_resp
                yield "\n"
            elif type(python_step_resp) is tuple:
                step_output.text = python_step_resp[0]
                step_output.data = python_step_resp[1]
                # vega_json = await vega_chart(step_output.data, uid)
                vega_json = None
                if not vega_json:
                    if type(step_output.data) is pd.DataFrame:
                        yield tabulate(
                            step_output.data.head(),
                            headers="keys",
                            tablefmt="pipe",
                            showindex=False,
                        )
                    elif type(step_output.data) is str:
                        yield step_output.data
                else:
                    yield "```vega\n"
                    yield vega_json
                    yield "\n```"
                yield "\n\n"
            elif type(python_step_resp) is pd.DataFrame:
                step_output.data = python_step_resp
                # vega_json = await vega_chart(step_output.data, uid)
                vega_json = None
                if not vega_json:
                    yield tabulate(
                        step_output.data.head(),
                        headers="keys",
                        tablefmt="pipe",
                        showindex=False,
                    )
                else:
                    yield "```vega\n"
                    yield vega_json
                    yield "\n```"
                yield "\n\n"
            else:
                raise NotImplementedError

        elif proc_step.type == "python_code":
            try:
                python_step_resp = await self.__run_python_code(
                    step_input, proc_step, uid
                )
            except Exception as e:
                raise e
            if type(python_step_resp) is str:
                step_output.text = python_step_resp
                yield python_step_resp
                yield "\n"
            elif type(python_step_resp) is tuple:
                step_output.text = python_step_resp[0]
                step_output.data = python_step_resp[1]
                vega_json = await vega_chart(step_output.data, uid)
                if not vega_json:
                    if type(step_output.data) is pd.DataFrame:
                        yield tabulate(
                            step_output.data.head(),
                            headers="keys",
                            tablefmt="pipe",
                            showindex=False,
                        )
                    elif type(step_output.data) is str:
                        yield step_output.data
                else:
                    yield "```vega\n"
                    yield vega_json
                    yield "\n```"
                yield "\n\n"
            elif type(python_step_resp) is pd.DataFrame:
                step_output.data = python_step_resp
                # vega_json = await vega_chart(step_output.data, uid)
                vega_json = None
                if not vega_json:
                    yield tabulate(
                        step_output.data.head(),
                        headers="keys",
                        tablefmt="pipe",
                        showindex=False,
                    )
                else:
                    yield "```vega\n"
                    yield vega_json
                    yield "\n```"
                yield "\n\n"
            else:
                raise NotImplementedError

        # Generate + run Python code
        elif proc_step.type == "code_generation":
            async for message in self.__run_llm_python_step(step_input, proc_step, uid):
                if type(message) is LLMResponse:
                    step_output = message
                else:
                    yield message
            yield "\n\n"

        # Nested BP step
        elif proc_step.type == "bp":
            try:
                sub_proc = get_s3_proc(proc_step.name, uid, settings.user_artifact_bucket)
            except Exception as e:
                raise IOError(f"Could not find sub-process: {proc_step.name}")
            try:
                async for message in self.run_proc(
                    step_input,
                    sub_proc,
                    uid,
                    True,
                    session_id=session_id,
                ):
                    if type(message) is LLMResponse:
                        step_output = message
                    else:
                        yield message
            except Exception as e:
                raise e

        elif proc_step.type == "rag":
            rag_messages = []
            try:
                async for message in self.__run_rag_step(step_input, proc_step, uid):
                    if type(message) is not FlowLog:
                        rag_messages.append(message)
                    yield message
            except Exception as e:
                raise e
            rag_response = "".join(rag_messages)
            step_output.text = rag_response

        elif proc_step.type == "semantic":
            semantic_search_messages = []
            try:
                async for message in self.__run_semantic_search_step(
                    step_input, proc_step, uid
                ):
                    if type(message) is not FlowLog:
                        semantic_search_messages.append(message)
                    yield message
                semantic_search_resp = "".join(semantic_search_messages)
                step_output.text = semantic_search_resp
            except Exception as e:
                raise e
            semantic_search_resp = "".join(semantic_search_messages)
            step_output.text = semantic_search_resp

        else:
            raise NotImplementedError(f"Unrecognized step type {proc_step.type}")

        yield step_output

    async def process_llm(
        self,
        question: str,
        proc_step: Optional[ProcessStep],
        uid: str,
        manifest: Dict = None,
        session_id: str = "",
        use_alias: Optional[bool] = False,
    ) -> AsyncGenerator:
        """Call the LLM (more documentation forthcoming)."""
        if use_alias:
            raise NotImplementedError

        # Load manifest
        if not manifest:
            manifest = copy.deepcopy(proc_step.manifest)
        else:
            manifest = copy.deepcopy(manifest)

        ##
        # Parse through data sources. There's currently three types supported:
        #     1. PostgreSQL database - pull the schema and prepend it to the prompt
        #     2. CSV file - load in the file as a pd.DataFrame, prepend the df.head() to the prompt
        #     3. Text file - prepend the contents of the file to the prompt
        #     4. Image file - add to "Image" field of manifest
        data_schemas = []
        sources = None
        if proc_step:
            sources = proc_step.data_sources

        if sources and sources[0] != "none":
            for data_source in sources:
                data_str = None
                data = get_data_from_s3(data_source, uid, settings.user_artifact_bucket)
                if type(data) is dict:
                    data = data.get("data")

                # CSV file
                if type(data) is StringIO:
                    try:
                        df_data = StringIO(data)
                        df = pd.read_csv(df_data)
                        data_str = f"FILENAME: {data_source}\n\n{df.to_string()}"
                    except Exception as e:
                        raise ValueError(
                            f"Got error {e} while attempting to load dataframe."
                        )

                # Text file
                elif type(data) is str and type(data) and not is_base64_image(data):
                    data_str = f"FILENAME: {data_source}\n\n{data}"

                # DataFrame
                elif type(data) is pd.DataFrame:
                    data_str = f"FILENAME: {data_source}\n\n{data.to_string()}"

                elif is_base64_image(data):
                    if manifest.get("images"):
                        manifest["images"].append(data)
                    else:
                        manifest["images"] = [data]

                else:
                    raise NotImplementedError(f"Unsupported data type {type(data)}")

                if data_str:
                    data_schemas.append(data_str)

            manifest["prompt"] = "\n".join(data_schemas) + f"\n{manifest['prompt']}"

        if is_base64_image(question):
            if manifest.get("images"):
                manifest["images"].append(proc_step.data.get(question))
            else:
                manifest["images"] = [question]

        # Initialize a process
        try:
            if proc_step:
                # Hack to allow both manifests and processes
                proc = proc_step.parent
            else:
                proc = get_s3_proc("Assistant", "SYSTEM", settings.user_artifact_bucket)

            # Create session (single-use) and add question
            session = ChatSession(
                proc.slashgpt_config,
                manifest=manifest,
                agent_name=proc.server_config.agent_name,
            )
            if not is_base64_image(question):
                session.append_user_question(session.manifest.format_question(question))
                yield FlowLog(
                    message=f"[Assistant][call_llm] Received question: {session.manifest.format_question(question)}"
                )

            # Call the LLM
            res, function_call = None, None
            retry_attempts = 0
            while retry_attempts < proc.server_config.max_retries and not res:
                if proc_step:
                    try:
                        if proc_step.debug_output_name and proc_step.manifest:
                            full_prompt = f"{manifest.get('prompt')}\n\n{question}"
                            _ = await upload_to_s3(
                                proc_step.debug_output_name, full_prompt, uid, settings.user_artifact_bucket
                            )
                    except Exception as e:
                        raise IOError(
                            f"Got error {e} while attempting to upload {proc_step.debug_output_name}"
                        )
                try:
                    res = None
                    # Non-streaming cases, like function calling or image generation
                    if (
                        manifest.get("stream") in [False, None]
                        or manifest.get("tool_code") is not None
                    ):
                        async for message in session.call_loop(
                            self._process_event, None
                        ):
                            if message:
                                res = message
                                yield message
                    # Regular streaming case
                    else:
                        collected_messages = []
                        async for message in session.call_loop(
                            self._process_event, None
                        ):
                            if type(message) is not FlowLog:
                                collected_messages.append(message)
                            yield message  # Stream output to UI
                        res = "".join(collected_messages)
                except Exception as e:
                    yield FlowLog(error=f"[Assistant][inference] Got error {e}")
                if not res:
                    retry_attempts += 1

            yield FlowLog(
                message=f"[Assistant][inference] Got response from LLM: {res}"
            )

        except (NoCredentialsError, PartialCredentialsError) as e:
            yield FlowLog(error=f"[Assistant][call_llm] AWS credential error")
        except Exception as e:
            yield FlowLog(
                error=f"[Assistant][call_llm] Uncaught error while making inference: {e}"
            )

    async def inference(
        self,
        question,
        uid: str,
        session_id: str = "",
    ) -> AsyncGenerator:
        """Manages LLM inference (more documentation forthcoming)."""

        yield FlowLog(
            message=f"[Assistant][inference] Received request from user: {uid}"
        )

        user_proc = None
        show_description = True
        try:
            # Override AI question dispatcher with dot-slash syntax
            try:
                aws_token_error = False
                if question.startswith("./"):
                    proc_name = question.split("--")[0].split("/")[1].strip()
                    if len(question.split("--")) > 1:
                        user_question = "".join(question.split("--")[1:]).strip()
                        question = user_question
                    try:
                        if uid == "SYSTEM":
                            show_description = False
                        user_proc = get_s3_proc(proc_name, uid, settings.user_artifact_bucket)
                    except (NoCredentialsError, PartialCredentialsError) as ce:
                        aws_token_error = True
                        async for chunk in generate_stream(
                            f"{proc_name}: Received error: {ce}"
                        ):
                            yield chunk
                    except GraphParseError as gp:
                        async for chunk in generate_stream(
                            f"{proc_name}: Received error: {gp}"
                        ):
                            yield chunk
                    # Could not find a matching process
                    if not user_proc and not aws_token_error:
                        async for chunk in generate_stream(
                            f"Could not find a process called {proc_name}"
                        ):
                            yield chunk

                else:
                    dummy_manifest = None
                    try:
                        dummy_proc = get_s3_proc("Assistant", "SYSTEM", settings.user_artifact_bucket)
                        dummy_manifest = dummy_proc.manifests.get("chat")
                    except Exception as e:
                        async for chunk in generate_stream(
                            f"Received error while pulling expert [Assistant]: {e}"
                        ):
                            yield chunk

                    if not dummy_manifest:
                        raise IOError(
                            f"Could not access [Assistant] manifest from bucket {settings.user_artifact_bucket}. Check AWS connection."
                        )

                    proc_s3_path = f"d/{uid}/proc"
                    proc_bucket_contents = get_s3_dir_summary(settings.user_artifact_bucket, proc_s3_path)

                    dummy_manifest["prompt"] = (
                        parse_s3_proc_data_no_regex(proc_bucket_contents)
                        + dummy_manifest["prompt"]
                    )
                    # Determine if user wants information about the system, or wants to run a process
                    stream_to_ui = True
                    collected_messages = []
                    try:
                        async for message in self.process_llm(
                            question,
                            None,
                            uid,
                            dummy_manifest,
                            session_id=session_id,
                        ):
                            if type(message) is FlowLog:
                                yield message
                            else:
                                collected_messages.append(message)
                                if collected_messages[0].startswith("`"):
                                    stream_to_ui = False
                                if stream_to_ui:
                                    yield message
                        if stream_to_ui:
                            yield "\n\n"
                    except Exception as e:
                        raise e

                    # Parse LLM response to detect if a process should run
                    llm_res = "".join(collected_messages)
                    proc_name = None
                    pattern = r"```([^`]*)```"
                    match = re.search(pattern, llm_res)
                    if match:
                        proc_name = match.group(1).strip()
                    if proc_name:
                        user_proc = get_s3_proc(proc_name, uid, settings.user_artifact_bucket)
                        # Could not find a matching process
                        if not user_proc:
                            raise ValueError(
                                f"Could not find an process called {proc_name}"
                            )
            except (IndexError, ValueError):
                async for chunk in generate_stream(
                    "Could not detect a valid process in your query. Please try again."
                ):
                    yield chunk
            except Exception as e:
                raise e

            if user_proc is not None:
                # Run the process and return those results
                yield FlowLog(
                    message=f"[Assistant][inference] Running process {user_proc.name}"
                )
                try:
                    # message: Tuple[Union[str, LLMResponse], bool]
                    async for message in self.run_proc(
                        question,
                        user_proc,
                        uid,
                        session_id=session_id,
                        show_description=show_description,
                    ):
                        if type(message) is tuple:
                            if (
                                type(message[0]) is not LLMResponse
                                and message[1] is True
                            ):
                                yield message[0]
                        else:
                            if type(message) is not LLMResponse:
                                yield message
                except Exception as e:
                    raise e

        except (NoCredentialsError, PartialCredentialsError) as e:
            yield FlowLog(
                error=f"[SlashGPTServer][refresh_from_s3] AWS credential error: {e}"
            )
            raise e
        except Exception as e:
            raise e

    # run_proc is a wrapper for run_proc_inner_impl.
    # Its purpose is to setup a new enclosing OTel span.
    # Only run_proc should call run_proc_inner_impl.
    async def run_proc(
        self,
        question: str,
        proc: BP,
        uid: str,
        is_sub_proc: bool = False,
        step_no: int = 0,
        session_id: str = "",
        show_description: bool = True,
    ) -> AsyncGenerator:
        tracer = trace.get_tracer(__name__)
        fun_name = "BPAgent.run_proc"
        span_name = f'{fun_name}("{proc.name}")'
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("tne.orchestration.fun_name", fun_name)
            span.set_attribute("tne.orchestration.proc_name", proc.name)
            async for message in self.run_proc_inner_impl(
                question,
                proc,
                uid,
                is_sub_proc=is_sub_proc,
                step_no=step_no,
                session_id=session_id,
                show_description=show_description,
            ):
                yield message

    # Only run_proc should call run_proc_inner_impl.
    async def run_proc_inner_impl(
        self,
        question: str,
        proc: BP,
        uid: str,
        is_sub_proc: bool = False,
        step_no: int = 0,
        session_id: str = "",
        show_description: bool = True,
    ) -> AsyncGenerator:
        self.orig_question = question
        step_input = question

        # Cache input here for dispatched steps
        dispatched_input = None

        # Error state
        had_error = False

        # Processes run in batch will always start at the beginning
        proc.current_step = step_no
        for i in range(step_no, len(proc)):
            if had_error:
                break
            step_output = None
            is_spinning = False
            proc_step = proc.steps[i]

            # If we errored on the previous step, break the loop
            if type(proc_step) is ProcessStep:
                yield FlowLog(
                    message=f"[BPAgent][run_proc] Running step: {proc_step.description}"
                )

                ##
                # Emit special token to tell client to generate a spinner for high latency tasks
                #     1. Function call
                #     2. Image generation
                ##
                if proc_step.manifest:
                    if not proc_step.manifest.get("model"):
                        is_spinning = True
                        yield "```SET_IS_SPINNING```"
                        yield "\n"
                    elif (
                        proc_step.manifest.get("model").get("model_name")
                        in image_models
                    ):
                        is_spinning = True
                        yield "```SET_IS_SPINNING```"
                        yield "\n"

                collected_messages = []
                try:
                    async for message in self.__run_step(
                        proc_step,
                        proc,
                        step_input,
                        dispatched_input,
                        uid,
                        session_id,
                        is_spinning,
                        show_description,
                    ):
                        if type(message) is tuple:
                            if type(message[0]) is str:
                                message = (
                                    replace_escaped_newlines(message[0]),
                                    message[1],
                                )
                            collected_messages.append(message[0])
                            yield message
                        else:
                            if type(message) is str:
                                message = replace_escaped_newlines(message)
                            collected_messages.append(message)
                            yield message, not proc_step.suppress_output
                except AttributeError as ae:
                    async for chunk in generate_stream(
                        f"{ae}. Please check if the file exists."
                    ):
                        yield chunk, True
                except NotImplementedError:
                    async for chunk in generate_stream(
                        f"\nMalformed process step {proc_step.name}. Please check your expert configuration and try again."
                    ):
                        yield chunk, True
                except Exception as e:
                    raise e

                step_output = collected_messages[-1]

            elif type(proc_step) is list:
                # Emit spinning token for parallel tasks
                is_spinning = True
                yield "```SET_IS_SPINNING```"
                yield "\n"

                tasks = []
                try:
                    for parallel_step in proc_step:
                        # Coroutine that collects messages from the async generator
                        collector = collect_messages(
                            self.__run_step(
                                parallel_step,
                                proc,
                                step_input,
                                dispatched_input,
                                uid,
                                session_id,
                                is_spinning,
                            )
                        )
                        tasks.append(collector)
                except Exception as e:
                    raise e

                # Run all collector coroutines concurrently and collect their results
                results = None
                try:
                    results = await asyncio.gather(*tasks)
                except AttributeError as ae:
                    async for chunk in generate_stream(
                        f"{ae}. Please check if the file exists."
                    ):
                        yield chunk, True  # FIXME(lucas): Allow canvas output management for parallel steps

                # Each item in results is a list of messages from one async generator
                if results:
                    llm_responses = []
                    for model_output in results:
                        llm_responses.append(model_output)

                    aggregate_str = ""
                    for resp in llm_responses:
                        aggregate_str += f"{resp}\n\n"
                    step_output = LLMResponse(text=aggregate_str, data=None)

            else:
                raise NotImplementedError

            # Save any data we've collected
            if step_output:
                proc_steps = proc_step if type(proc_step) is list else [proc_step]
                for s in proc_steps:
                    data_filename = s.data_output_name
                    if data_filename:
                        if data_filename.endswith(".csv"):
                            data_payload = {
                                "type": "dataframe",
                                "description": f"Data collected by step {s.name}",
                                "data": step_output.data,
                            }
                        else:
                            data_payload = {
                                "type": "text",
                                "description": f"Data collected by step {s.name}",
                                "data": step_output.text,
                            }
                        proc.data[s.data_output_name.split(".")[0]] = data_payload
                        # Uploads step data to S3 data bucket
                        try:
                            if (
                                data_filename.endswith(".csv")
                                or data_filename.endswith(".png")
                                or data_filename.endswith(".jpg")
                                or data_filename.endswith(".jpeg")
                            ):
                                await upload_to_s3(
                                    s.data_output_name, step_output.data, uid, settings.user_artifact_bucket
                                )
                            else:
                                if proc_step.type != "code_generation":
                                    await upload_to_s3(
                                        s.data_output_name, step_output.text, uid, settings.user_artifact_bucket
                                    )
                            yield FlowLog(
                                message=f"[BPAgent][run_proc] Uploaded {s.data_output_name} to S3..."
                            )
                        except Exception as e:
                            raise IOError(
                                f"[BPAgent][run_proc] Got error {e} uploading {s.data_output_name} to S3"
                            )

                # Update inputs for the next step
                if type(proc_step) is list:
                    proc_step = proc_step[0]

                if proc_step.output_type == "augment_prompt":
                    # Add the step output to the prompt
                    step_str = ""
                    if step_output.text is not None:
                        step_str += f"\n\n{step_output.text}"
                    if step_output.data is not None:
                        newline_str = ""
                        if len(step_str) > 0:
                            newline_str = "\n\n"
                        if type(step_output.data) is pd.DataFrame:
                            step_str += f"{newline_str}{step_output.data.head()}"
                        elif type(step_output.data) is str:
                            step_str += f"{newline_str}{step_output.data}"
                    step_input += step_str
                elif proc_step.output_type == "dispatch":
                    # Load the next manifest from possible_outputs dictionary
                    dispatched_input = copy.deepcopy(step_input)
                    step_input = proc_step.possible_outputs.get(step_output.text)
                else:
                    step_str = ""
                    # This is an error message
                    if type(step_output) is str:
                        had_error = True
                    else:
                        if step_output.text:
                            step_str += f"{step_output.text}"
                        if step_output.data is not None and not is_base64_image(
                            step_output.data
                        ):
                            newline_str = ""
                            if len(step_str) > 0:
                                newline_str += "\n\n"
                            if type(step_output.data) is pd.DataFrame:
                                step_str += f"{newline_str}{step_output.data.head()}"
                            elif type(step_output.data) is str:
                                step_str += f"{newline_str}{step_output.data}"
                        if step_output.data is not None:
                            step_input = step_output.data
                        else:
                            step_input = step_output.text

                # Send token to stop spinning
                if is_spinning:
                    yield "\n"
                    yield "```STOP_SPINNING```"
                    yield "\n"

                # Stream data to client
                if i == len(proc) - 1 or step_no == len(proc) - 1:
                    if not is_sub_proc and type(step_output) is LLMResponse:
                        if type(step_output.data) is pd.DataFrame:
                            yield tabulate(
                                step_output.data,
                                headers="keys",
                                tablefmt="pipe",
                                showindex=False,
                            )
                            yield "\n\n"
                        else:
                            if type(step_output) is LLMResponse:
                                if step_output.data is not None:
                                    if (
                                        proc_step.manifest.get("model").get(
                                            "model_name"
                                        )
                                        in image_models
                                    ):
                                        yield step_output.text
                                        yield "\n\n"
                                    elif not is_base64_image(step_output.data):
                                        yield step_output.data
                                        yield "\n\n"
                    else:
                        yield step_output
                        yield "\n\n"

    async def __run_llm_step(
        self,
        step_input: Union[str, pd.DataFrame],
        proc_step: ProcessStep,
        uid: str,
        session_id: str = "",
    ) -> AsyncGenerator:
        retry_no = 0
        llm_resp = None

        while retry_no < proc_step.parent.server_config.max_retries and not llm_resp:
            collected_messages = []
            try:
                # FIXME(rakuto): Hack for TNE BigText Model routing. Dispatching to TNE models should be integrated into Slash-GPT.
                if proc_step.manifest:
                    if proc_step.manifest.get("model"):
                        model = proc_step.manifest.get("model")
                        if (
                            proc_step.manifest.get("model").get("model_name")
                            == "tne-bigtext-arxiv"
                        ):
                            endpoint = os.getenv(
                                "BIGTEXT_ENDPOINT", "http://localhost:5002/v1"
                            )
                            client = AsyncOpenAI(base_url=endpoint)
                            event_stream = await client.chat.completions.create(
                                model="bigtext-arxiv-endor",
                                messages=[{"role": "user", "content": step_input}],
                                stream=True,
                            )
                            async for event in event_stream:
                                content = event.choices[0].delta.content or ""
                                collected_messages.append(content)
                                yield content

                            llm_resp = "".join(collected_messages)
                            retry_no += 1
                        else:
                            async for message in self.process_llm(
                                question=step_input,
                                proc_step=proc_step,
                                uid=uid,
                                session_id=session_id,
                                use_alias=False,
                            ):
                                if type(message) is not FlowLog:
                                    collected_messages.append(message)
                                yield message
                            llm_resp = "".join(collected_messages)
                            retry_no += 1
            except Exception as e:
                raise e

    async def __run_rag_step(
        self, step_input: str, proc_step: ProcessStep, uid: str, session_id: str = ""
    ) -> AsyncGenerator:
        # Create a RagRequest from the step input
        if proc_step.rag_db_name:
            rag_request = create_rag_request(step_input, proc_step.rag_db_name)
        else:
            rag_request = create_rag_request(step_input)

        response_count = 0
        start_seconds = time.time()

        def log_progress():
            elapsed_seconds = time.time() - start_seconds
            return FlowLog(
                message=f"[RAG] response_count: {response_count:4d}, elapsed_seconds: {elapsed_seconds:7.3f}",
            )

        async def on_response(response: RagResponse, response_str: str):
            nonlocal response_count
            nonlocal start_seconds
            response_count += 1
            return (
                log_progress()
            )  # TODO(lucas): Probably don't need to log every packet, don't forget about this

        async def on_error(error_code: int, error_str: str):
            yield FlowLog(
                error="[RAG] Error code: {error_code}, Error string: {error_str}"
            )

        # Call the RAG service
        collected_responses = []
        is_spinning = True
        emitted_evaluations = False
        emitted_debug_header = False
        yield "```SET_IS_SPINNING```"
        yield "\n"
        try:
            resp_count = 0
            rag_start_time = time.time()
            evaluation_texts = {}
            async for rag_response in async_iterate_streaming_request_generator(
                settings.rag_endpoint, rag_request, on_response, on_error
            ):
                patch_record = rag_response.patch_record
                collected_responses.append(patch_record)
                if proc_step.show_debug is True:
                    if not emitted_debug_header:
                        yield "**RAG METRICS**\n\n"
                        emitted_debug_header = True
                    if patch_record.anns:
                        anns = patch_record.anns
                        if is_spinning:
                            is_spinning = False
                            yield "```STOP_SPINNING```"
                            yield "\n"
                        for embedding_id in anns.keys():
                            text = anns[embedding_id].text
                            similarity = anns[embedding_id].similarity
                            evaluation = anns[embedding_id].evaluation
                            if text:
                                yield f"EMBEDDING ID: {embedding_id}\n\n"
                                yield f"EXTRACTED TEXT:\n\n"
                                yield f"{text}\n\n"
                            if similarity:
                                yield f"EMBEDDING ID: {embedding_id}\n\n"
                                yield f"SIMILARITY SCORE: {similarity}\n\n"
                            if evaluation:
                                if evaluation.text:
                                    if evaluation_texts.get(embedding_id):
                                        evaluation_texts[
                                            embedding_id
                                        ] += evaluation.text
                                    else:
                                        evaluation_texts[embedding_id] = evaluation.text

                    if patch_record.events and proc_step.show_debug:
                        do_not_emit = [
                            "Completed: generate_full_history_summary",
                            "Completed: generate_prev_record_summary",
                            "Completed: generate_rag_output",
                            "Completed: generate_anns_summary",
                        ]
                        for event in patch_record.events:
                            if event.message not in do_not_emit:
                                yield f"RAG EVENT: {event.message}\nSEVERITY: {event.severity}\nSOURCE: {event.source}\n\n"

                    if patch_record.metrics and proc_step.show_debug:
                        metrics = patch_record.metrics
                        yield f"METRICS:\n\n"
                        yield f"ann_count_after_retrival: {metrics.ann_count_after_retrieval}\n"
                        yield f"ann_count_after_similarity_min_value: {metrics.ann_count_after_similarity_min_value}\n"
                        yield f"ann_count_after_relevancy_min_value: {metrics.ann_count_after_relevancy_min_value}\n"
                        yield f"ann_count_after_relevancy_max_count: {metrics.ann_count_after_relevancy_max_count}\n\n"

                if patch_record.rag_output:
                    # First check if there are any evaluations to emit
                    if not emitted_evaluations and proc_step.show_debug is True:
                        for embedding_id in evaluation_texts.keys():
                            yield f"EMBEDDING ID: {embedding_id}\n\n"
                            yield f"EVALUATION:\n\n"
                            yield f"{evaluation_texts[embedding_id]}\n\n"
                        emitted_evaluations = True
                        yield "**END RAG METRICS**\n\n"

                    if is_spinning:
                        is_spinning = False
                        yield "```STOP_SPINNING```"
                        yield "\n"

                    yield rag_response.patch_record.rag_output.text

                # Temporary logging
                resp_count += 1
                if resp_count % 25 == 0:
                    elapsed_time = time.time() - rag_start_time
                    yield FlowLog(
                        message=f"[RAG] response_count: {resp_count:4d}, elapsed_seconds: {elapsed_time:7.3f}",
                    )

        except Exception as e:
            yield FlowLog(
                error="Error running RAG. Please try again.",
            )
            if is_spinning:
                yield "```STOP_SPINNING```"
                yield "\n"
            raise e

    async def __run_semantic_search_step(
        self, step_input: str, proc_step: ProcessStep, uid: str, session_id: str = ""
    ) -> AsyncGenerator:
        # Create a RagRequest from the step input
        configs = {}
        if proc_step.rag_db_name:
            configs["rag_db_name"] = proc_step.rag_db_name
        if proc_step.max_count:
            configs["max_count"] = proc_step.max_count
        if proc_step.min_similarity:
            configs["min_similarity"] = proc_step.min_similarity

        anns_request = create_anns_request(step_input, **configs)

        # Call the service
        is_spinning = True
        yield "```SET_IS_SPINNING```"
        yield "\n"
        try:
            anns_request_bytes = anns_request_to_json_str(anns_request).encode()
            response_obj = requests.post(
                settings.anns_endpoint,
                data=anns_request_bytes,
                headers={"Content-Type": "application/json"},
            )
            status_code = response_obj.status_code

            if status_code == 200:
                if is_spinning:
                    is_spinning = False
                    yield "```STOP_SPINNING```"
                    yield "\n"

                response = anns_response_from_json_str(response_obj.content.decode())
                if is_spinning:
                    is_spinning = False
                    yield "```STOP_SPINNING"
                    yield "\n"
                for i, ann in enumerate(response.anns):
                    response_str = ""
                    if ann.similarity:
                        response_str += (
                            f"**Embedding #{i + 1}** (Similarity: {ann.similarity})\n\n"
                        )
                    if ann.text:
                        response_str += f"**Text**\n{ann.text}\n\n"
                    if ann.sources:
                        response_str += f"**Sources**\n"
                        for source in ann.sources:
                            response_str += f"  * {source.file_path}\n"
                        response_str += f"\n"
                    if ann.evaluation:
                        response_str += f"**Evaluation**\n{ann.evaluation}\n"
                    if ann.relevancy:
                        response_str += f"**Relevancy**\n{ann.relevancy}\n"
                    response_str += "\n"
                    yield response_str
            else:
                if is_spinning:
                    yield "```STOP_SPINNING```"
                    yield "\n"
                if status_code == 500:
                    yield "No results returned from your search. The query may not be relevant to the selected document corpus, but you may also try lowering the Similarity Threshold."
                else:
                    yield FlowLog(
                        error=f"ERROR {status_code}: Error running semantic search. Please try again.",
                    )

        except Exception as e:
            if is_spinning:
                yield "```STOP_SPINNING```"
                yield "\n"
            raise e

    async def __run_llm_python_code(
        self, llm_code, step, step_input, uid, retry_no, session_id
    ) -> AsyncGenerator:
        """Code interpreter module; allow the LLMs to generate and run Python code on the data within the BP."""
        if llm_code:
            # Save the LLM-generated code to a temporary file
            temp_file_path = save_code_to_temp_file(llm_code)

            # Install the TNE Python SDK into the code execution environment
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", TNE_PACKAGE_PATH]
                )
            except subprocess.CalledProcessError as e:
                raise e

            try:
                # Execute the temporary file and capture the namespace
                namespace = execute_temp_file(temp_file_path)

                # Access the results from the namespace
                df_last_3_months = namespace.get("df_last_3_months")
                result = namespace.get("result")

                if df_last_3_months is not None:
                    logger.debug(f"df_last_3_months: {df_last_3_months.head()}")
                    logger.debug(f"df_last_3_months shape: {df_last_3_months.shape}")
                else:
                    logger.error("df_last_3_months not found in namespace")

            except Exception as e:
                logger.error(f"Exception occurred: {e}")
                result = f"# USER QUERY: {step_input}\n\nresult = None"
                yield result
            finally:
                # Clean up the temporary file
                os.remove(temp_file_path)

            yield result

    async def __run_llm_python_step(
        self, step_input: Any, proc_step: ProcessStep, uid: str, session_id: str = ""
    ) -> AsyncGenerator:
        """Generate and run Python code that operates on DataFrames"""
        # Generate the code
        retry_no = 0
        llm_resp = None

        # 1. Use TNE Python SDK package to inject relevant data into LLM prompt
        data_context_buffer = f"UID: {uid}\nBUCKET: {settings.user_artifact_bucket}\n"
        session = TNE(uid, settings.user_artifact_bucket)

        # a. Inject data from graph UI
        for s in proc_step.data_sources:
            data_context_buffer = update_data_context_buffer(
                session, s, data_context_buffer
            )

        # b. If step input exists, inject it into data_context_buffer
        if type(step_input) is pd.DataFrame:
            step_input = step_input.head().to_string()
        elif type(step_input) is str:
            if len(step_input) >= DATA_BUFFER_LENGTH:
                step_input = step_input[:DATA_BUFFER_LENGTH]
        data_context_buffer += f"PROCESS_INPUT: {step_input}"

        # API call to the LLM for code generation
        code_gen_proc = get_s3_proc("CodeGen", "SYSTEM", settings.user_artifact_bucket)
        code_gen_manifest = code_gen_proc.manifests.get("codeGenerator")
        if proc_step.model_name:
            code_gen_manifest["model"] = {
                "engine_name": proc_step.engine_name,
                "model_name": proc_step.model_name,
                "api_key": proc_step.api_key,
            }
        code_gen_prompt = f"{data_context_buffer}\n\n{code_gen_manifest}"
        if proc_step.prompt:
            code_gen_prompt = (
                f"{code_gen_prompt}\n\nPROMPT FROM USER: {proc_step.prompt}"
            )
        code_gen_manifest["prompt"] = code_gen_prompt
        while retry_no < proc_step.parent.server_config.max_retries and not llm_resp:
            collected_messages = []
            async for message in self.process_llm(
                question=step_input,
                proc_step=None,
                manifest=code_gen_manifest,
                uid=uid,
                session_id=session_id,
                use_alias=False,
            ):
                if type(message) is not FlowLog:
                    collected_messages.append(message)
                yield message
            yield "\n\n"
            llm_resp = "".join(collected_messages)
            retry_no += 1

        # Execute the code
        regex_pattern = "```"
        parsed_resp = self.__parse_llm_response(llm_resp, regex_pattern)
        if type(parsed_resp) is str:
            if "python" in parsed_resp:
                formatted_code = parsed_resp.split("python\n")[1]
                formatted_code = f"# USER QUERY: {step_input}\n\n{formatted_code}"
            else:
                formatted_code = parsed_resp
                formatted_code = f"# USER QUERY: {step_input}\n\n{formatted_code}"
        elif (
            type(parsed_resp) is FlowLog
            and parsed_resp.message
            == "[Assistant][call_llm] Regex pattern ``` match not detected. Returning unfiltered output."
        ):
            formatted_code = llm_resp
            formatted_code = f"# USER QUERY: {step_input}\n\n{formatted_code}"
        # The LLM didn't generate code; likely because of a conversational, non-data question
        else:
            formatted_code = f"# USER QUERY: {step_input}\n\nresult = 'None'"

        # Error case
        if type(formatted_code) is FlowLog:
            yield formatted_code

        # Remove notebook-like syntax with result variable
        if re.search(r"\n\nresult$", formatted_code):
            formatted_code = re.sub(r"\n\nresult$", "", formatted_code)

        # Run the generated code
        collected_messages = []
        async for message in self.__run_llm_python_code(
            formatted_code, proc_step, step_input, uid, 0, session_id
        ):
            collected_messages.append(message)

            if type(message) is not pd.DataFrame and type(message) is not pd.Series:
                yield str(message)

        if proc_step.data_output_name:
            await upload_to_s3(proc_step.data_output_name, formatted_code, uid, settings.user_artifact_bucket)

        ret = collected_messages[-1]
        if type(ret) is pd.DataFrame:
            yield LLMResponse(text=formatted_code, data=ret)
        elif type(ret) is pd.Series:
            yield LLMResponse(text=formatted_code, data=pd.DataFrame(ret))
        elif type(ret) is str:
            yield LLMResponse(text=ret)
        else:
            yield LLMResponse(text=formatted_code, data=str(ret))

    async def __run_python_code(
        self,
        step_input: Optional[Any],
        proc_step: ProcessStep,
        uid: str,
        session_id: str = "",
    ) -> Any:
        module_name, module_code = fetch_python_module(proc_step.name, uid, settings.user_artifact_bucket)

        # Step input is available through special variable PROCESS_INPUT
        namespace = {"PROCESS_INPUT": step_input}

        # Install the TNE Python SDK into the code execution environment
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", TNE_PACKAGE_PATH]
            )
        except subprocess.CalledProcessError as e:
            return {"error": str(e)}

        try:
            exec(f'__name__ = "__main__"\n{module_code}', {}, namespace)
        except Exception as e:
            raise e

        return namespace.get("result")

    # DEPRECATED - current experts use __run_python_code
    async def __run_python_step(
        self,
        step_input: Union[str, pd.DataFrame],
        proc_step: ProcessStep,
        uid: str,
        session_id: str = "",
    ) -> Any:
        namespace = {}
        func_name = proc_step.name.split(".")[0]

        # If any SQL queries, run them and collect the dataframe results
        sql_kwargs = {}
        if proc_step.sql_queries:
            # Construct the function scope for running SQL
            sql_namespace = {}
            run_sql_name, run_sql_code = get_python_s3_module("run_sql", "SYSTEM", settings.user_artifact_bucket)
            exec(run_sql_code, sql_namespace)

            # Iteratively run each SQL query
            run_sql = sql_namespace[run_sql_name.split(".")[0]]
            for query_name in proc_step.sql_queries.keys():
                sql_query = proc_step.sql_queries[query_name]
                try:
                    df = await run_sql(
                        sql_query,
                        proc=proc_step.parent,
                        db_name="ebp",
                    )
                    if len(df) > 1:
                        sql_kwargs[query_name] = df[1]
                except Exception as e:
                    raise ValueError(
                        f"Got Postgres error {e} running query {sql_query}"
                    )

        try:
            # Load Python code from S3
            proc_step.name = proc_step.name.split(".")[0]
            module_name, module_code = get_python_s3_module(proc_step.name, uid, settings.user_artifact_bucket)
            if re.search(r"^FEATURE_FLAG_KNATIVE_RUNTIME = True$", module_code, re.M):
                # Add kwargs from the step graph
                kwargs = proc_step.kwargs

                rt = krt.KnativeRuntime()
                name = module_name.replace("_", "-").replace(".py", "") + "-" + uid[0:6]
                ret = await rt.aexec(
                    name=name,
                    inputs=kwargs,
                    # endpoint="http://localhost:8081",
                )
                return ret
            else:
                exec(module_code, namespace)
        except krt.ExecutionError as e:
            return FlowLog(error="Remote code execution failed", exc_info=e)
        except (NoCredentialsError, PartialCredentialsError, Exception) as e:
            return FlowLog(
                error=f"[SlashGPTServer][__run_python_step] AWS credentials error: {e}"
            )
        # Construct the function scope
        if func_name in namespace and callable(namespace[func_name]):
            func = namespace[func_name]
            func_params = inspect.signature(func).parameters

            func_scope = {}
            sources = proc_step.data_sources
            if sources and sources[0] != "none":
                for data_source in sources:
                    try:
                        session = TNE(uid, settings.user_artifact_bucket)
                        proc_step.data.update(
                            {
                                data_source.split(".")[0].strip(): session.get_object(
                                    data_source
                                )
                            }
                        )
                    except Exception as e:
                        pass
                for k in proc_step.data.keys():
                    if len(k.split(".")) > 1:
                        key_name = k.split(".")[0]
                    else:
                        key_name = k
                    func_scope[key_name] = proc_step.data.get(k)

            # Handle kwargs
            kwargs = {}
            proc_kwargs = proc_step.kwargs

            # Add kwargs from the step graph
            if proc_kwargs:
                kwargs.update(proc_kwargs)

            # Add any SQL queries to the kwargs
            kwargs.update(sql_kwargs)

            for param in func_params:
                ##
                # Special case: some functions may need to access process-level object
                # like database connectors, etc. and as such functions may have a BP
                # as an argument. This looks for that case and fills out the parent process
                ##
                if func_params.get(param).annotation == BP:
                    kwargs.update({param: proc_step.parent})

            # Merge func_scope and kwargs
            if len(func_scope) > 0:
                kwargs.update(func_scope)

            # Case 1: this step has no kwargs
            try:
                if len(kwargs) == 0:
                    # Async functions
                    if inspect.iscoroutinefunction(func):
                        step_output = await func(step_input, session_id=session_id)
                    # Sync functions
                    else:
                        step_output = func(step_input, session_id)
                # Case 2: this step has kwargs
                else:
                    # Async functions
                    if inspect.iscoroutinefunction(func):
                        if proc_step.use_prev_input is not False:
                            step_output = await func(
                                step_input,
                                session_id=session_id,
                                **kwargs,
                            )
                        else:
                            step_output = await func(
                                step_input, session_id=session_id, **kwargs
                            )
                    # Sync functions
                    else:
                        step_output = func(step_input, session_id=session_id, **kwargs)
            except Exception as e:
                raise e

            return step_output

        return None

    @classmethod
    def __parse_llm_response(cls, res, pattern) -> Union[str, FlowLog]:
        # Filter the LLM output based off of a specified regex pattern
        match = None
        if pattern:
            re_pattern = r"{}(.*?){}".format(re.escape(pattern), re.escape(pattern))
            match = re.search(re_pattern, res, re.DOTALL)
        else:
            return FlowLog(
                message="[Assistant][call_llm] No regex pattern specified. No pattern matching performed."
            )

        # Try to extract a list if the pattern is found
        if match:
            formatted_res = match.group(1).strip()
            try:
                # FIXME(lucas): this is stupid
                if (
                    "[" in formatted_res
                    and "import" not in formatted_res
                    and "Announcement" not in formatted_res
                    and "json" not in formatted_res
                ):
                    formatted_res = formatted_res.strip("[]").split(",")
                    formatted_res = [i.strip() for i in formatted_res]
                return formatted_res
            except Exception as e:
                return FlowLog(
                    error=f"[Assistant][call_llm] Detected malformed list in LLM response. Likely LLM hallucination."
                )
        else:
            return FlowLog(
                message=f"[Assistant][call_llm] Regex pattern {pattern} match not detected. Returning unfiltered output."
            )
