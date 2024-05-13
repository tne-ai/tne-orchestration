import os
import re
import uuid
import yaml
import boto3
import random
import asyncio

import base64
import binascii
from orchestration.bp import BP
from io import StringIO


from typing import Optional
from collections import deque
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# RAG imports
from orchestration.v2.api.api import (
    EmbeddingsConfig,
    LlmConfig,
    RagConfig,
    RagRecord,
    RagRequest,
)

from typing import Union, Dict, Tuple

# Inference server literals
BUCKET_NAME = "bp-authoring-files"
PROC_DIR = "proc"
AGENT_DIR = "manifests"
CODE_DIR = "modules"
DATA_DIR = "data"
OPERATOR_NODES = ["llm", "proc", "python", "rag"]

# RAG literals
RAG_DB_HOST = "postgresql-ebp.cfwmuvh4blso.us-west-2.rds.amazonaws.com"


def __random_uuid():
    return str(uuid.uuid4())


def __get_rag_config(rag_db_name):
    """Construct RAGConfig object (currently not configurable)"""
    embeddings_db_pass = os.getenv("EMBEDDINGS_DB_PASSWORD")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    shared_llm_config = LlmConfig(model="gpt-4-0125-preview", api_key=openai_api_key)

    return RagConfig(
        embeddings_config=EmbeddingsConfig(
            db_host=RAG_DB_HOST,
            db_port=5432,
            db_name=rag_db_name,
            db_username="postgres",
            db_password=embeddings_db_pass,
            model="text-embedding-ada-002",
            api_key=openai_api_key,
        ),
        anns_input_llm_config=shared_llm_config,
        ann_evaluations_llm_config=shared_llm_config,
        ann_relevancies_llm_config=shared_llm_config,
        anns_summary_llm_config=shared_llm_config,
        prev_record_summary_llm_config=shared_llm_config,
        full_history_summary_llm_config=shared_llm_config,
        rag_output_llm_config=shared_llm_config,
    )


def create_rag_request(
    user_input: str,
    rag_db_name: str = "crag_agent_db",
    config: Optional[RagConfig] = None,
    history: Optional[list] = None,
):
    if not config:
        config = __get_rag_config(rag_db_name)
    if not history:
        history = []
    # Random IDs for now
    request_id = __random_uuid()
    thread_id = __random_uuid()

    # Create a new RagRecord and RagRequest using the user input
    new_record = RagRecord(request_id=request_id, config=config, user_input=user_input)
    request = RagRequest(thread_id=thread_id, history=history, new_record=new_record)

    return request


def is_base64_png(s):
    """Determines if a serialized string represents a base64 encoded PNG"""
    if type(s) is not str:
        return False

    try:
        s += "=" * ((4 - len(s) % 4) % 4)
        decoded_bytes = base64.b64decode(s, validate=True)
    except (ValueError, binascii.Error):
        return False

    # Check for PNG signature
    png_signature = b"\x89PNG\r\n\x1a\n"
    return decoded_bytes.startswith(png_signature)


def is_base64_jpg(s):
    """Determines if a serialized string represents a base64 encoded JPEG"""
    if type(s) is not str:
        return False

    try:
        s += "=" * ((4 - len(s) % 4) % 4)  # Ensure padding is correct
        decoded_bytes = base64.b64decode(s, validate=True)
    except (ValueError, binascii.Error):
        return False

    # Check for JPG signature (JPEG files start with FF D8)
    jpg_signature = b"\xFF\xD8"
    return decoded_bytes.startswith(jpg_signature)


def is_base64_image(s):
    return is_base64_jpg(s) or is_base64_png(s)


async def generate_stream(text):
    """Fakes a streaming response for consistency"""
    words = text.split()
    for ind, word in enumerate(words):
        await asyncio.sleep(random.uniform(0.06, 0.09))
        if ind < len(words) - 1:
            yield f"{word} "
        else:
            yield f"{word}\n"


def get_s3_dir_summary(bucket_name: str, prefix: str) -> str:
    """Summarize a S3 directory (for feeding downstream into LLM prompt)"""
    s3 = boto3.client("s3")
    dir_summary = ""  # Initialize the output string

    # List objects within the specified bucket and prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" in response:
        for obj in response["Contents"]:
            file_name = obj["Key"]

            # Get the object from S3
            file_obj = s3.get_object(Bucket=bucket_name, Key=file_name)

            # Read the file content
            file_content = file_obj["Body"].read().decode("utf-8")

            # Append the file name and its content to the output string
            dir_summary += f"{file_name}\n\n{file_content}\n"
            # Adding an extra newline for separation between files in output
            dir_summary += "-" * 40 + "\n\n"

    return dir_summary


def get_s3_ls(bucket_name: str, prefix: str) -> str:
    """List the contents of a S3 bucket"""
    s3 = boto3.client("s3")
    dir_summary = ""  # Initialize the output string

    # List objects within the specified bucket and prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" in response:
        for obj in response["Contents"]:
            file_name = obj["Key"]

            # Append the file name and its content to the output string
            dir_summary += f"{file_name}\n"
            # Adding an extra newline for separation between files in output
            dir_summary += "-" * 40 + "\n\n"

    return dir_summary


def get_python_s3_module(module_name, uid):
    """Load Python code from S3"""
    try:
        s3 = boto3.client("s3")
        python_s3_path = f"d/{uid}/{CODE_DIR}"
        bucket_contents = s3.list_objects(Bucket=BUCKET_NAME, Prefix=python_s3_path)[
            "Contents"
        ]

        # Load processes from S3
        for obj in bucket_contents:
            basename = obj.get("Key").split("/")[-1].split(".")[0]
            if basename == module_name:
                file_name = f"{basename}.py"
                file_content = (
                    s3.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])["Body"]
                    .read()
                    .decode("utf-8")
                )
                return file_name, file_content

    except (NoCredentialsError, PartialCredentialsError) as e:
        raise e
    except Exception as e:
        raise e

    return None


def upload_to_s3(file_name, data, uid) -> str:
    """Upload an object to S3"""
    s3_path = f"d/{uid}/{DATA_DIR}"

    # Handle CSV upload by using StringIO
    if file_name.endswith(".csv"):
        csv_buffer = StringIO()
        data.to_csv(csv_buffer, index=False)
        s3_data = csv_buffer.getvalue()
        content_type = "text/csv"
    elif file_name.endswith(".png"):
        s3_data = data
        content_type = "image/png"
    elif file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
        s3_data = data
        content_type = "image/jpeg"
    else:
        s3_data = data
        content_type = "text/plain"

    full_file_name = f"{s3_path}/{file_name}"

    try:
        # Create an S3 client
        s3 = boto3.client("s3")

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=full_file_name,
            Body=s3_data,
            ContentType=content_type,
        )

        region = s3.meta.region_name
        s3_url = f"https://{BUCKET_NAME}.s3.{region}.amazonaws.com/{full_file_name}"

        return s3_url
    except Exception as e:
        raise e


def get_s3_proc(proc_name: str, uid: str):
    """Fetches a process file from S3"""
    try:
        s3 = boto3.client("s3")
        proc_s3_path = f"d/{uid}/{PROC_DIR}"
        bucket_contents = s3.list_objects(Bucket=BUCKET_NAME, Prefix=proc_s3_path)[
            "Contents"
        ]

        # Load processes from S3
        for obj in bucket_contents:
            s3_proc = obj.get("Key").split("/")[-1].split(".")[0]
            if s3_proc == proc_name:
                file_content = s3.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])[
                    "Body"
                ].read()
                if file_content:
                    proc_contents = parse_graph(s3_proc, yaml.safe_load(file_content))
                    proc = BP(proc_contents, uid)
                    return proc

    except (NoCredentialsError, PartialCredentialsError) as e:
        raise e
    except Exception as e:
        raise e

    return None


def parse_s3_proc_data(text):
    """
    Parses the given text to extract information about processes, including operators and data nodes.

    :param text: The input text containing process information.
    :return: A formatted string with the parsed data.
    """
    output = ""

    # Pattern to match "proc name" followed by nodes within the same proc
    proc_pattern = re.compile(
        r"d/\d+/proc/([^\n]+)\n\nnodes:\n(.*?)(?=\nd/|$)", re.DOTALL
    )

    # Find all proc sections
    proc_matches = proc_pattern.findall(text)
    num_procs = len(proc_matches)

    # Include the count of procs found at the beginning
    output += f"Total Processes Available: {num_procs}\n\n"

    for proc_match in proc_matches:
        proc_name, nodes_section = proc_match
        output += f"Process Name: {proc_name.strip()}\n  - Operators:\n"

        # Extract information for operator nodes (llm and python)
        operator_pattern = re.compile(
            r"type: (llm|python)\n.*?title: (.*?)\n.*?(?:slashgptManifest: (.*?)\n|module: (.*?)\n)",
            re.DOTALL,
        )
        operator_matches = operator_pattern.findall(nodes_section)

        for operator_match in operator_matches:
            node_type, title, manifest, module = operator_match
            if node_type == "llm":
                output += f"      - Operator Type: llm, Title: {title.strip()}, Manifest: {manifest.strip()}\n"
            elif node_type == "python":
                output += f"      - Operator Type: python, Title: {title.strip()}, Module: {module.strip()}\n"

        output += "  - Data:\n"

        # Extract information for data nodes (csv, text, image)
        data_pattern = re.compile(
            r"type: (csv|text|image)\n.*?title: (.*?)\n.*?(?:csvFile: (.*?)\n|textFile: (.*?)\n|imageFile: (.*?)\n)",
            re.DOTALL,
        )
        data_matches = data_pattern.findall(nodes_section)

        for data_match in data_matches:
            node_type, title, csv_file, text_file, image_file = data_match
            file_name = csv_file or text_file or image_file
            output += (
                f"    - Data Type: {title.strip()}, Filename: {file_name.strip()}\n"
            )

    output += "---\n"

    # print(output)
    return output


def parse_s3_proc_data_no_regex(text):
    """
    Parses the given text without using regular expressions to extract information about processes,
    including operators and data nodes, and includes the count of procs found at the beginning of the output.

    :param text: The input text containing process information.
    :return: A formatted string with the parsed data.
    """
    lines = text.split("\n")
    proc_name = ""
    in_nodes_section = False
    num_procs = 0
    operators = []
    data = []
    output = ""

    for line in lines:
        if line.startswith("d/"):
            if proc_name:  # Output the previous proc before starting a new one
                output += format_proc(proc_name, operators, data)
                operators, data = [], []  # Reset for the next proc
            proc_name = line.split("/proc/")[1]
            num_procs += 1
            in_nodes_section = False
        elif line.strip() == "nodes:":
            in_nodes_section = True
        elif in_nodes_section and line.startswith("  - id:"):
            # Reset flags when a new node starts
            current_type = current_title = current_manifest = current_module = (
                current_file_name
            ) = ""
        elif "type:" in line:
            current_type = line.split(":")[1].strip()
        elif "title:" in line:
            current_title = line.split(":", 1)[1].strip()
        elif "slashgptManifest:" in line:
            current_manifest = line.split(":", 1)[1].strip()
        elif "module:" in line:
            current_module = line.split(":", 1)[1].strip()
        elif any(x in line for x in ["csvFile:", "textFile:", "imageFile:"]):
            current_filename = line.split(":", 1)[1].strip()
            # Data node complete
            if current_type in ["csv", "text", "image"]:
                data.append({"title": current_title, "filename": current_filename})
        elif "outputType:" in line:
            # Operator node complete
            if current_type in ["llm", "python", "rag"]:
                operators.append(
                    {
                        "type": current_type,
                        "title": current_title,
                        "manifest": current_manifest,
                        "module": current_module,
                    }
                )

    # Output the last proc
    if proc_name:
        output += format_proc(proc_name, operators, data)

    # Prepend the total number of procs found
    output = f"Total Processes Available: {num_procs}\n\n" + output
    return output


def format_proc(proc_name, operators, data):
    """
    Formats a single proc's information.

    :param proc_name: The name of the process.
    :param operators: A list of operator nodes.
    :param data: A list of data nodes.
    :return: A formatted string of the process information.
    """
    proc_output = f"Proc Name: {proc_name}\n  - Operators:\n"
    for op in operators:
        type_info = f"Type: {op['type']}, Title: {op['title']}"
        if op["type"] in ["llm", "rag"]:  # Treat 'rag' similar to 'llm'
            manifest = op.get("manifest", "")
            type_info += f", Manifest: {manifest}"
        elif op["type"] == "python":
            module = op.get("module", "")
            type_info += f", Module: {module}"
        proc_output += f"      - {type_info}\n"
    proc_output += "  - Data:\n"
    for d in data:
        proc_output += f"    - Title: {d['title']}, Filename: {d['filename']}\n"
    proc_output += "---\n"
    return proc_output


def __get_node(node_id, nodes):
    """Extract a node object from a process graph"""
    for node in nodes:
        if node["id"] == node_id:
            return node
    return None


def __construct_step_dict(
    node, step_input, data_sources, data_output_name, debug_output_name, sql_queries
) -> Union[Dict, Tuple]:
    """Construct a graph node object given I/O information and node data"""

    # Process node
    if node.get("type") == "proc":
        step_dict = {
            "name": node.get("data").get("proc").split(".")[0],
            "type": "bp",
            "input": step_input,
            "description": node.get("data").get("title"),
            "output_type": node.get("data").get("outputType"),
            "suppress_output": node.get("data").get("outputToCanvas"),
            "data_sources": data_sources,
        }
        if data_output_name:
            step_dict["data_output_name"] = data_output_name

        return step_dict

    # Agent node
    elif node.get("type") == "llm":
        step_dict = {
            "name": node.get("data").get("slashgptManifest"),
            "type": "llm",
            "description": node.get("data").get("title"),
            "output_type": node.get("data").get("outputType"),
            "suppress_output": node.get("data").get("outputToCanvas"),
            "tool_json": node.get("data").get("toolJson"),
            "tool_code": node.get("data").get("toolCode"),
            "data_sources": data_sources,
            "input": step_input,
        }
        if data_output_name:
            step_dict["data_output_name"] = data_output_name
        if debug_output_name:
            step_dict["debug_output_name"] = debug_output_name
        if step_dict.get("name") == "dall-e-3":
            step_dict["stream"] = False

        # Handle map and passthrough logic
        if node.get("data").get("outputType") == "dispatch":
            step_dict["possible_outputs"] = {}
            step_input = node.get("data").get("input")
            for line in step_input.split("\n"):
                class_name, child_manifest = line.split(":")
                step_dict["possible_outputs"][
                    class_name.strip()
                ] = child_manifest.strip()

            # Creating the 'dispatched' step
            dispatched_step = {
                "name": "dispatched",
                "type": "llm",
                "output_type": "overwrite",
                "suppress_output": node.get("data").get(
                    "outputToCanvas"
                ),  # Inherit outputToCanvas from parent
                "description": "Conditional logic to determine next step based on difficulty classification.",
            }

            return step_dict, dispatched_step
        else:
            return step_dict

    # Code node
    elif node.get("type") == "python":
        kwargs = {}
        graph_kwargs = node.get("data").get("kwargs")
        if len(graph_kwargs) > 0:
            for line in graph_kwargs.split("\n"):
                kwarg, value = line.split(":")
                kwargs[kwarg.strip()] = value.strip()

        step_dict = {
            "name": node.get("data").get("module"),
            "type": "python",
            "description": node.get("data").get("title"),
            "output_type": node.get("data").get("outputType"),
            "suppress_output": node.get("data").get("outputToCanvas"),
            "data_sources": data_sources,
            "kwargs": kwargs,
            "input": step_input,
        }

        if data_output_name:
            step_dict["data_output_name"] = data_output_name
        if sql_queries:
            step_dict["sql_queries"] = sql_queries

        return step_dict

    # RAG node
    elif node.get("type") == "rag":
        step_dict = {
            "name": "rag",
            "type": "rag",
            "description": node.get("data").get("title"),
            "output_type": node.get("data").get("outputType"),
            "suppress_output": node.get("data").get("outputToCanvas"),
            "show_debug": node.get("data").get("showDebug"),
            "rag_db_name": node.get("data").get("ragDbName"),
            "data_sources": data_sources,
            "input": step_input,
        }
        if data_output_name:
            step_dict["data_output_name"] = data_output_name

        return step_dict


def __get_step_io_data(incoming_nodes, child_nodes, nodes):
    """Check graph edges to get I/O information about a particular node"""
    sql_queries = {}
    data_sources = []
    step_input, data_output_name, debug_output_name = None, None, None

    # Nodes for which current node is the target
    for n in incoming_nodes:
        incoming_node = __get_node(n, nodes)
        if incoming_node.get("type") == "database":
            if incoming_node.get("data").get("queryName"):
                sql_queries[incoming_node.get("data").get("queryName")] = (
                    incoming_node.get("data").get("query")
                )
            else:
                data_sources.append(incoming_node.get("data").get("dbName"))
        elif incoming_node.get("type") == "csv":
            data_name = incoming_node.get("data").get("csvFile")
            data_sources.append(f"{data_name}.csv")
        elif incoming_node.get("type") == "text":
            data_name = incoming_node.get("data").get("textFile")
            data_sources.append(f"{data_name}.txt")
        elif incoming_node.get("type") == "query":
            step_input = incoming_node.get("data").get("input")
        elif incoming_node.get("type") == "image":
            data_name = f"{incoming_node.get('data').get('imageFile')}"
            data_sources.append(data_name)

    # Targets of the current node
    for c in child_nodes:
        outgoing_node = __get_node(c, nodes)
        if outgoing_node.get("type") == "out":
            data_output_name = outgoing_node.get("data").get("outputName")
        if outgoing_node.get("type") == "debug":
            debug_output_name = outgoing_node.get("data").get("outputName")

    return step_input, data_sources, data_output_name, debug_output_name, sql_queries


def parse_graph(file_name, graph):
    """Construct a process graph dictionary that the bp-runner server can parse"""

    # Build a set of all incoming operator nodes
    incoming_from_proc = {
        edge["target"]
        for edge in graph["edges"]
        if __get_node(edge["source"], graph["nodes"])["type"] in OPERATOR_NODES
        and __get_node(edge["target"], graph["nodes"])["type"] in OPERATOR_NODES
    }

    # Find root node: one of the operator nodes without any incoming operators
    start_ids = []
    for node in graph["nodes"]:
        if node["type"] in OPERATOR_NODES and node["id"] not in incoming_from_proc:
            start_ids.append(node["id"])

    if len(start_ids) == 0:
        return None

    start_id = start_ids[0]
    visited = set()
    skip_nodes = set()
    queue = deque([start_id])

    proc_file = {
        "name": file_name,
        "description": "Placeholder description",
        "data": {
            "ebp": {
                "type": "PostgreSQL",
                "description": "Main EBP database",
                "creds": {
                    "user": "DB_USER",
                    "password": "DB_PASS",
                    "host": "DB_HOST",
                    "port": "DB_PORT",
                },
            },
        },
        "output_type": "overwrite",
        "data_output_name": "",
        "steps": {},
    }

    steps = []
    nodes = graph.get("nodes")
    while queue:
        node_id = queue.popleft()

        if node_id in visited or node_id in skip_nodes:
            continue

        node = __get_node(node_id, nodes)
        if not node:
            continue

        visited.add(node_id)
        child_nodes = [
            edge["target"] for edge in graph["edges"] if edge["source"] == node_id
        ]
        incoming_nodes = [
            edge["source"] for edge in graph["edges"] if edge["target"] == node_id
        ]

        (
            step_input,
            data_sources,
            data_output_name,
            debug_output_name,
            sql_queries,
        ) = __get_step_io_data(incoming_nodes, child_nodes, nodes)

        child_operators = []
        for n in child_nodes:
            outgoing_node = __get_node(n, nodes)
            if (
                outgoing_node.get("type") in OPERATOR_NODES
                and node.get("data").get("outputType") != "dispatch"
            ):
                child_operators.append(n)

        step_info = __construct_step_dict(
            node,
            step_input,
            data_sources,
            data_output_name,
            debug_output_name,
            sql_queries,
        )

        ##
        # Scatter/gather graph construction
        # If multiple LLMs need to be run in parallel, we create a list of ProcessStep objects for the server to parse.
        # We need to handle things a bit differently between if the scatter operation is the first step vs. intermediate
        ##

        # Case 1: Scatter/gather as first operation
        parallel_step_info = []
        append_parallel_steps = False
        if len(start_ids) > 1 and node_id == start_id:
            child_parents = [
                edge["source"]
                for edge in graph["edges"]
                if edge["target"] == child_operators[0]
            ]

            parallel_step_info = []
            for op in child_parents:
                op_node = __get_node(op, nodes)
                (
                    op_input,
                    op_data,
                    op_data_output,
                    op_debug_output_name,
                    op_sql_queries,
                ) = __get_step_io_data(incoming_nodes, child_nodes, nodes)
                op_step_dict = __construct_step_dict(
                    op_node,
                    op_input,
                    op_data,
                    op_data_output,
                    op_debug_output_name,
                    op_sql_queries,
                )
                parallel_step_info.append(op_step_dict)
                skip_nodes.update(child_parents)
                for edge in graph["edges"]:
                    if (
                        edge["source"] == child_parents[0]
                        and edge["target"] not in visited
                    ):
                        queue.append(edge["target"])

            step_info = parallel_step_info

        # Case 2: Intermediate scatter/gather
        if len(child_operators) > 1:
            for op in child_operators:
                op_node = __get_node(op, nodes)
                op_incoming_nodes = [node_id]
                op_children = [
                    edge["target"] for edge in graph["edges"] if edge["source"] == op
                ]
                (
                    op_input,
                    op_data,
                    op_data_output,
                    op_debug_output_name,
                    op_sql_queries,
                ) = __get_step_io_data(op_incoming_nodes, op_children, nodes)
                op_step_dict = __construct_step_dict(
                    op_node,
                    op_input,
                    op_data,
                    op_data_output,
                    op_debug_output_name,
                    op_sql_queries,
                )
                parallel_step_info.append(op_step_dict)
                skip_nodes.update(child_operators)
                for edge in graph["edges"]:
                    if (
                        edge["source"] == child_nodes[0]
                        and edge["target"] not in visited
                    ):
                        queue.append(edge["target"])
                append_parallel_steps = True

        ##
        # Serial implementation graph construction. Just one LLM being run.
        ##
        if type(step_info) is dict:
            steps.append(step_info)
            for edge in graph["edges"]:
                if edge["source"] == node_id and edge["target"] not in visited:
                    queue.append(edge["target"])

        # Handle graph traversal update in scatter/gather case
        elif type(step_info) is list:
            if len(start_ids) == 1:
                skip_nodes.update(child_nodes)
            steps.append(step_info)

        # Handle graph traversal update in map/passthrough case
        elif type(step_info) is tuple:
            # Handle the dispatcher LLM step
            step_dict = step_info[0]
            steps.append(step_dict)

            # Next step is already constructed in map/passthrough, so add it, and skip traversal to grandchild node
            dispatched_step = step_info[1]
            skip_nodes.update(child_nodes)
            steps.append(dispatched_step)
            for edge in graph["edges"]:
                if edge["source"] == child_nodes[0] and edge["target"] not in visited:
                    queue.append(edge["target"])

        if append_parallel_steps:
            steps.append(parallel_step_info)

        for edge in graph["edges"]:
            if edge["source"] == node_id and edge["target"] not in visited:
                queue.append(edge["target"])

    proc_file["steps"] = steps

    return proc_file
