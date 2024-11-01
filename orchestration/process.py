import asyncio
import base64
import copy

# Temporary function call workarounds to be incorporated into SlashGPT
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
from typing import Any, AsyncGenerator, Dict, Union, List, Optional

import boto3
import pandas as pd
import requests
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from opentelemetry import trace
from orchestration.bp import BP, ProcessStep
from orchestration.server_utils import (
    generate_stream,
    get_s3_proc,
    get_s3_ls,
    get_s3_dir_summary,
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
from tabulate import tabulate
from tne.TNE import TNE


if settings.use_local_slashgpt:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../SlashTNE/src"))
from slashgpt.chat_session import ChatSession

logger = logging.getLogger(__name__)
BUFFER_LENGTH = 20000
CTX_LENGTH = 1000

# Literal constants
TNE_PACKAGE_PATH = "./tne-0.0.1-py3-none-any.whl"
DATA_DIR = "Data"
image_models = ["dall-e-3"]

smr_client = boto3.client("sagemaker-runtime")  # type: SageMakerRuntimeClient

if platform.system() == "Darwin":
    # So that input can handle Kanji & delete
    import readline  # noqa: F401


class _Base(BaseModel):
    pass


class GraphParseError(Exception):
    def __init__(self, message):
        super().__init__(message)


class CodeRunError(Exception):
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
        if type(data) is pd.DataFrame:
            data_context_buffer += f"{file_name}\n\n{data.head().to_string()}\n\n"
        elif type(data) is dict:
            data_context_buffer += f"Multi-sheet excel file: {file_name}\n\n"
            for k in data.keys():
                data_context_buffer += f"   Sheet name: {k}\n\n{data[k].head()}\n\n"
        elif type(data) is str:
            if len(data) <= BUFFER_LENGTH:
                data_context_buffer += f"{file_name}\n\n{data}\n\n"
            else:
                data_context_buffer += (
                    f"CONTEXT FOR {file_name}\n\n{data[:BUFFER_LENGTH]}\n\n"
                )
        else:
            data_context_buffer += f"{data[:BUFFER_LENGTH]}\n\n"
    except ValueError:
        data = session.get_object_bytes(file_name).decode("utf-8")
        data_context_buffer += f"{data}]\n\nn"
    except IOError:
        return data_context_buffer

    return data_context_buffer


def execute_temp_file(file_path: str, env_vars: dict) -> dict:
    exec_namespace = env_vars.copy()
    result_namespace = runpy.run_path(file_path, init_globals=exec_namespace)
    return result_namespace


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


class BPAgent:
    def __init__(self, callback=None):
        self._callback = callback or self._noop

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
        is_spinning,
        show_description=True,
        project: Optional[str] = None,
        version: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator:
        tracer = trace.get_tracer(__name__)
        fun_name = "BPAgent.run_step"
        span_name = f'{fun_name}("{proc_step.type}", "{proc_step.name}")'

        # Disable project description text
        if project or version:
            show_description = False

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
                show_description,
                project=project,
                version=version,
                history=history,
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
        show_description=True,
        project: Optional[str] = None,
        version: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator:
        # TEMPORARY: route all tne-branded models to groq
        if proc_step.manifest:
            manifest_model = proc_step.manifest.get("model")
            if manifest_model:
                manifest_model.get("model_name")

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
                        step_input,
                        proc_step,
                        uid,
                        project=project,
                        version=version,
                        history=history,
                    ):
                        if type(message) is not FlowLog:
                            collected_messages.append(message)
                        yield message
                except Exception as e:
                    raise e
                llm_step_output = "".join(collected_messages)
                regex_pattern = "```"
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
                        step_input,
                        proc_step,
                        uid,
                        project=project,
                        version=version,
                        history=history,
                    ):
                        if type(message) is FlowLog:
                            yield message
                        else:
                            llm_step_messages.append(message)

                    img_url = "".join(llm_step_messages)

                    # Generate a unique filename based off of the image contents
                    data_s3_path = f"d/{uid}/{DATA_DIR}"
                    data_filenames = get_s3_ls(
                        settings.user_artifact_bucket, data_s3_path
                    )

                    datetime = str(time.time()).split(".")
                    img_filename = f"{datetime[0]}-{datetime[1]}.png"

                    response = requests.get(img_url)
                    if response.status_code == 200:
                        try:
                            img_s3_url = await upload_to_s3(
                                img_filename,
                                response.content,
                                uid,
                                settings.user_artifact_bucket,
                                project=project,
                                version=version,
                            )
                        except Exception as e:
                            raise e
                        step_output.text = f"![]({img_s3_url})"
                        step_output.data = base64.b64encode(response.content).decode(
                            "utf-8"
                        )
                        yield step_output.text
                        yield FlowLog(
                            message=f"[BPAgent][run_proc] Uploaded {img_filename} to S3..."
                        )
                    else:
                        yield FlowLog(
                            error="[BPAgent][run_proc] Failed to generate image..."
                        )
                        raise IOError("Failed to generate image")
                except Exception as e:
                    raise e
            else:
                try:
                    async for message in self.__run_llm_step(
                        step_input,
                        proc_step,
                        uid,
                        project=project,
                        version=version,
                        history=history,
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
                parsed_resp = self.__parse_llm_response(llm_step_output, regex_pattern)
                formatted_output = parsed_resp if parsed_resp else llm_step_output
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
                    step_input, proc_step, uid, project=project, version=version
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
                if type(step_output.data) is pd.DataFrame:
                    yield tabulate(
                        step_output.data.head(),
                        headers="keys",
                        tablefmt="pipe",
                        showindex=False,
                    )
                elif type(step_output.data) is str:
                    yield step_output.data
                yield "\n\n"
            elif type(python_step_resp) is pd.DataFrame:
                step_output.data = python_step_resp
                yield tabulate(
                    step_output.data.head(),
                    headers="keys",
                    tablefmt="pipe",
                    showindex=False,
                )
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
                if type(step_output.data) is pd.DataFrame:
                    yield tabulate(
                        step_output.data.head(),
                        headers="keys",
                        tablefmt="pipe",
                        showindex=False,
                    )
                elif type(step_output.data) is str:
                    yield step_output.data
                yield "\n\n"
            elif type(python_step_resp) is pd.DataFrame:
                step_output.data = python_step_resp
                yield tabulate(
                    step_output.data.head(),
                    headers="keys",
                    tablefmt="pipe",
                    showindex=False,
                )
                yield "\n\n"
            else:
                raise NotImplementedError

        # Generate + run Python code
        elif proc_step.type == "code_generation":
            retries = 0
            done = False
            while retries <= settings.max_retries and not done:
                try:
                    async for message in self.__run_llm_python_step(
                        step_input,
                        proc_step,
                        uid,
                        project=project,
                        version=version,
                        history=history,
                    ):
                        if type(message) is LLMResponse:
                            step_output = message
                        else:
                            yield message
                        done = True
                except CodeRunError as ce:
                    logger.warning(ce)
                    retries += 1
                    if retries > settings.max_retries:
                        done = True
                        raise IOError(
                            "Code generation failed. Please reword your question and try again."
                        )

                yield "\n\n"

        # Nested BP step
        elif proc_step.type == "bp":
            try:
                sub_proc = get_s3_proc(
                    proc_step.name, uid, settings.user_artifact_bucket
                )
            except Exception:
                raise IOError(f"Could not find sub-process: {proc_step.name}")
            try:
                async for message in self.run_proc(
                    step_input,
                    sub_proc,
                    uid,
                    is_sub_proc=True,
                    project=project,
                    version=version,
                    history=history,
                ):
                    if type(message) is tuple:
                        if type(message[0]) is LLMResponse:
                            step_output = message[0]
                        else:
                            yield message
                    else:
                        yield message
            except Exception as e:
                raise e

        elif proc_step.type == "rag":
            rag_messages = []
            try:
                async for message in self.__run_rag_step(step_input, proc_step):
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
                    step_input,
                    proc_step,
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
        use_alias: Optional[bool] = False,
        project: Optional[str] = None,
        version: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = False,
    ) -> AsyncGenerator:
        """Call the LLM (more documentation forthcoming)."""
        if use_alias:
            raise NotImplementedError

        # Load manifest
        if not manifest:
            manifest = copy.deepcopy(proc_step.manifest)
        else:
            manifest = copy.deepcopy(manifest)

        # Set random seed in manifest
        manifest["seed"] = settings.random_seed

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
                data = get_data_from_s3(
                    data_source,
                    uid,
                    settings.user_artifact_bucket,
                    project=project,
                    version=version,
                )
                if type(data) is dict:
                    data = data.get("data")
                if data is None:
                    raise ValueError(f"Data source {data_source} is null")

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

        # FIXME(lucas): Hardcoded
        if manifest.get("model").get("model_name") == "echo-chat":
            manifest["model"]["engine_name"] = "ollama"
            manifest["model"]["api_key"] = "OPENAI_API_KEY"
            manifest["prompt"] = ""

        # Initialize a process
        try:
            if proc_step:
                # Hack to allow both manifests and processes
                proc = proc_step.parent
            else:
                proc = get_s3_proc(
                    "Assistant",
                    "SYSTEM",
                    settings.user_artifact_bucket,
                    project=project,
                    version=version,
                )

            # Create session (single-use) and add question
            session = ChatSession(
                proc.slashgpt_config,
                manifest=manifest,
                agent_name=proc.server_config.agent_name,
            )
            if not is_base64_image(question):
                if history:
                    for msg in history:
                        session.append_message(
                            role=msg.get("role"),
                            message=msg.get("content"),
                            preset=False,
                        )
                session.append_user_question(session.manifest.format_question(question))
                yield FlowLog(
                    message=f"[Assistant][call_llm] Received question: {session.manifest.format_question(question)}"
                )

            # Call the LLM
            res, _function_call = None, None
            retry_attempts = 0
            while retry_attempts < proc.server_config.max_retries and not res:
                if proc_step:
                    try:
                        if proc_step.debug_output_name and proc_step.manifest:
                            full_prompt = f"{manifest.get('prompt')}\n\n{question}"
                            try:
                                _ = await upload_to_s3(
                                    proc_step.debug_output_name,
                                    full_prompt,
                                    uid,
                                    settings.user_artifact_bucket,
                                    project=project,
                                    version=version,
                                )
                            except Exception as e:
                                raise e
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

        except (NoCredentialsError, PartialCredentialsError):
            yield FlowLog(error="[Assistant][call_llm] AWS credential error")
        except Exception as e:
            yield FlowLog(
                error=f"[Assistant][call_llm] Uncaught error while making inference: {e}"
            )

    async def inference(
        self,
        question,
        uid: str,
        project: Optional[str],
        version: Optional[str],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator:
        """Manages LLM inference (more documentation forthcoming)."""

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
                        user_proc = get_s3_proc(
                            proc_name=proc_name,
                            uid=uid,
                            bucket_name=settings.user_artifact_bucket,
                            project=project,
                            version=version,
                        )
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
                        dummy_proc = get_s3_proc(
                            "Assistant", "SYSTEM", settings.user_artifact_bucket
                        )
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
                    proc_bucket_contents = get_s3_dir_summary(
                        settings.user_artifact_bucket, proc_s3_path
                    )

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
                        user_proc = get_s3_proc(
                            proc_name,
                            uid,
                            settings.user_artifact_bucket,
                            project=project,
                            version=version,
                        )
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
                        project=project,
                        version=version,
                        history=history,
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
        is_sub_proc: Optional[bool] = False,
        project: Optional[str] = None,
        version: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        step_no: int = 0,
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
                step_no=step_no,
                history=history,
                project=project,
                version=version,
                is_sub_proc=is_sub_proc,
                show_description=show_description,
            ):
                yield message

    # Only run_proc should call run_proc_inner_impl.
    async def run_proc_inner_impl(
        self,
        question: str,
        proc: BP,
        uid: str,
        is_sub_proc: Optional[bool] = False,
        project: Optional[str] = None,
        version: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        step_no: int = 0,
        show_description: bool = True,
    ) -> AsyncGenerator:
        step_input = question

        # Cache input here for dispatched steps
        orig_question = question
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

                if proc_step.use_user_query is True:
                    step_input = f"[Context query: {orig_question}] [Current input: {step_input}]"

                collected_messages = []
                try:
                    async for message in self.__run_step(
                        proc_step,
                        proc,
                        step_input,
                        dispatched_input,
                        uid,
                        is_spinning,
                        show_description,
                        project=project,
                        version=version,
                        history=history,
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
                            if type(message) is str and not is_base64_image(message):
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

                if type(step_output) is LLMResponse:
                    if step_output.data is not None:
                        if proc_step.manifest:
                            if (
                                not proc_step.manifest.get("model").get("model_name")
                                in image_models
                            ):
                                yield FlowLog(
                                    message=f"[BPAgent][run_proc] Output for {proc_step.description}: {str(step_output.data)}"
                                )
                        else:
                            if not is_base64_image(step_output.data):
                                yield FlowLog(
                                    message=f"[BPAgent][run_proc] Output for {proc_step.description}: {str(step_output.data)}"
                                )
                    elif step_output.text:
                        yield FlowLog(
                            message=f"[BPAgent][run_proc] Output for {proc_step.description}: {step_output.text}"
                        )
                        if proc_step.manifest:
                            if (
                                proc_step.manifest.get("model").get("model_name")
                                in image_models
                            ):
                                yield step_output.text

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
                                is_spinning,
                                project=project,
                                version=version,
                                history=history,
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
                if is_sub_proc:
                    yield step_output
                proc_steps = proc_step if type(proc_step) is list else [proc_step]
                for s in proc_steps:
                    output_files = s.output_files
                    if output_files:
                        for output_file in output_files:
                            ext = output_file.split(".")[-1]
                            # Uploads step data to S3 data bucket
                            try:
                                if (
                                    ext in ["csv", "png", "jpg", "jpeg", "xlsx"]
                                    and step_output.data is not None
                                ):
                                    try:
                                        await upload_to_s3(
                                            output_file,
                                            step_output.data,
                                            uid,
                                            settings.user_artifact_bucket,
                                            project=project,
                                            version=version,
                                        )
                                    except Exception as e:
                                        raise e
                                else:
                                    if step_output.text:
                                        try:
                                            await upload_to_s3(
                                                output_file,
                                                step_output.text,
                                                uid,
                                                settings.user_artifact_bucket,
                                                project=project,
                                                version=version,
                                            )
                                        except Exception as e:
                                            raise e
                                yield FlowLog(
                                    message=f"[BPAgent][run_proc] Uploaded {output_file} to S3..."
                                )
                            except Exception as e:
                                raise IOError(
                                    f"[BPAgent][run_proc] Got error {e} uploading {output_file} to S3"
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
                        elif type(step_output.data) is str and not is_base64_image(step_output.data):
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

    async def __run_llm_step(
        self,
        step_input: Union[str, pd.DataFrame],
        proc_step: ProcessStep,
        uid: str,
        project: Optional[str] = None,
        version: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator:
        retry_no = 0
        llm_resp = None

        while retry_no < proc_step.parent.server_config.max_retries and not llm_resp:
            collected_messages = []
            try:
                # FIXME(rakuto): Hack for TNE BigText Model routing. Dispatching to TNE models should be integrated into Slash-GPT.
                # TODO(lucas): Chat history for hosted models
                if proc_step.manifest:
                    if proc_step.manifest.get("model"):
                        async for message in self.process_llm(
                            question=step_input,
                            proc_step=proc_step,
                            uid=uid,
                            use_alias=False,
                            project=project,
                            version=version,
                            history=history,
                        ):
                            if type(message) is not FlowLog:
                                collected_messages.append(message)
                            yield message
                        llm_resp = "".join(collected_messages)
                        retry_no += 1
            except Exception as e:
                raise e

    async def __run_rag_step(
        self,
        step_input: str,
        proc_step: ProcessStep,
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
            return log_progress()

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
                                yield "EXTRACTED TEXT:\n\n"
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
                        yield "METRICS:\n\n"
                        yield f"ann_count_after_retrival: {metrics.ann_count_after_retrieval}\n"
                        yield f"ann_count_after_similarity_min_value: {metrics.ann_count_after_similarity_min_value}\n"
                        yield f"ann_count_after_relevancy_min_value: {metrics.ann_count_after_relevancy_min_value}\n"
                        yield f"ann_count_after_relevancy_max_count: {metrics.ann_count_after_relevancy_max_count}\n\n"

                if patch_record.rag_output:
                    # First check if there are any evaluations to emit
                    if not emitted_evaluations and proc_step.show_debug is True:
                        for embedding_id in evaluation_texts.keys():
                            yield f"EMBEDDING ID: {embedding_id}\n\n"
                            yield "EVALUATION:\n\n"
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
        self,
        step_input: str,
        proc_step: ProcessStep,
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
                        response_str += "**Sources**\n"
                        for source in ann.sources:
                            response_str += f"  * {source.file_path}\n"
                        response_str += "\n"
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
        self,
        llm_code: str,
        step_input: any,
        uid: str,
        project: Optional[str] = None,
        version: Optional[str] = None,
    ) -> AsyncGenerator:
        """Code interpreter module; allow the LLMs to generate and run Python code on the data within the BP."""
        if llm_code:
            # Save the LLM-generated code to a temporary file
            temp_file_path = save_code_to_temp_file(llm_code)

            # Install the TNE Python SDK into the code execution environment
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--force-reinstall",
                        TNE_PACKAGE_PATH,
                    ]
                )
            except subprocess.CalledProcessError as e:
                raise e

            try:
                # Execute the temporary file and capture the namespace
                env_vars = {
                    "PROCESS_INPUT": step_input,
                    "BUCKET": settings.user_artifact_bucket,
                    "UID": uid,
                    "PROJECT": project,
                    "VERSION": version,
                }
                namespace = execute_temp_file(temp_file_path, env_vars)

                # Access the results from the namespace
                result = namespace.get("result")

                if result is None:
                    i = 0
                    temp_hardcoded_keys = [
                        "tote_range_segments",
                        "tote_sales_percentage",
                        "newness_range_monthly_sales",
                        "tote_contribution",
                        "rising_styles",
                        "clutch_range",
                        "backpack_range_selling_well_june",
                        "top_crossbody_bags"
                    ]
                    while result is None:
                        temp_key = temp_hardcoded_keys[i]
                        result = namespace.get(temp_key)
                        if result is not None:
                            result = result.to_pandas()
                        i += 1

            except Exception as e:
                raise CodeRunError(e)
            finally:
                # Clean up the temporary file
                os.remove(temp_file_path)

            yield result

    async def __run_llm_python_step(
        self,
        step_input: Any,
        proc_step: ProcessStep,
        uid: str,
        project: Optional[str] = None,
        version: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator:
        """Generate and run Python code that operates on DataFrames"""
        # Generate the code
        retry_no = 0
        llm_resp = None

        # 1. Use TNE Python SDK package to inject relevant data into LLM prompt
        session = TNE(
            uid, settings.user_artifact_bucket, project=project, version=version
        )

        # a. Inject data from graph UI
        data_context_buffer = ""
        for s in proc_step.data_sources:
            data_context_buffer = update_data_context_buffer(
                session, s, data_context_buffer
            )

        # b. If step input exists, inject it into data_context_buffer
        if type(step_input) is pd.DataFrame:
            step_input = step_input.head().to_string()
        data_context_buffer += f"PROCESS_INPUT: {step_input}"

        # API call to the LLM for code generation
        code_gen_proc = get_s3_proc("CodeGen", "SYSTEM", settings.user_artifact_bucket)
        code_gen_manifest = code_gen_proc.manifests.get("codeGenerator")
        if proc_step.model_name:
            if proc_step.model_name == "echo-ie":
                code_gen_manifest["model"] = {
                    "engine_name": "ollama",
                    "model_name": proc_step.model_name,
                    "api_key": proc_step.api_key,
                }
            else:
                code_gen_manifest["model"] = {
                    "engine_name": proc_step.engine_name,
                    "model_name": proc_step.model_name,
                    "api_key": proc_step.api_key,
                }
        if code_gen_manifest.get("model").get("engine_name") != "ollama":
            code_gen_prompt = (
                f"{data_context_buffer}\n\n{code_gen_manifest.get('prompt')}"
            )
            if proc_step.prompt:
                code_gen_prompt = (
                    f"{code_gen_prompt}\n\nPROMPT FROM USER: {proc_step.prompt}"
                )
            code_gen_manifest["prompt"] = code_gen_prompt
        else:
            code_gen_manifest["prompt"] = ""
        while retry_no < proc_step.parent.server_config.max_retries and not llm_resp:
            collected_messages = []
            async for message in self.process_llm(
                question=step_input,
                proc_step=None,
                manifest=code_gen_manifest,
                history=history,
                uid=uid,
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
            elif "polsars" in parsed_resp:
                formatted_code = parsed_resp.split("polsars\n")[1]
            else:
                formatted_code = parsed_resp

        # The LLM didn't generate code; likely because of a conversational, non-data question
        else:
            formatted_code = llm_resp
        yield FlowLog(message=f"[BPAgent][run_proc] Generated code: {formatted_code}")

        # Run the generated code
        collected_messages = []
        try:
            async for message in self.__run_llm_python_code(
                formatted_code, step_input, uid, project=project, version=version
            ):
                collected_messages.append(message)

                if type(message) is not pd.DataFrame and type(message) is not pd.Series:
                    yield str(message)
        except CodeRunError as ce:
            raise ce

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
        project: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Any:
        module_name, module_code = fetch_python_module(
            proc_step.name,
            uid,
            settings.user_artifact_bucket,
            project=project,
            version=version,
        )

        # Step input is available through special variable PROCESS_INPUT
        namespace = {
            "PROCESS_INPUT": step_input,
            "BUCKET": settings.user_artifact_bucket,
            "UID": uid,
            "PROJECT": project,
            "VERSION": version,
        }

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

    @classmethod
    def __parse_llm_response(cls, res, pattern) -> Union[str, FlowLog]:
        # First look for python markdown specifically
        code_block_match = re.search(r"```python(.*?)```", res, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
        polars_block_match = re.search(r"```polsars(.*?)```", res, re.DOTALL)
        if polars_block_match:
            return polars_block_match.group(1).strip()
        else:
            PYTHON_TAG_LENGTH = 14
            python_tag_ind = res.find("<|python_tag|>")
            if python_tag_ind > -1:
                code_block_match = res[(python_tag_ind + PYTHON_TAG_LENGTH):]
                return code_block_match

        # Look for other markdown if not found
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
            except Exception:
                return FlowLog(
                    error="[Assistant][call_llm] Detected malformed list in LLM response. Likely LLM hallucination."
                )
        else:
            return None
