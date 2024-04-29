from v2.etl.embedder import EmbedderId
from v2.etl.json_util import read_json, write_json
from v2.etl.typer_util import typer_arg, typer_opt


def tokenize_and_embed(
    input_json_file: typer_arg(str, "TODO: Add help text."),
    output_json_file: typer_arg(str, "TODO: Add help text."),
    embedder_id: typer_opt(
        EmbedderId, "TODO: Add help text."
    ) = EmbedderId.openai_ada_002,
):
    """TODO: Add help text."""

    doc_obj = read_json(input_json_file)

    doc_obj["command_args"]["tokenize_and_embed_args"] = {
        "input_json_file": input_json_file,
        "output_json_file": output_json_file,
        "embedder_id": embedder_id.name,
    }

    embedder_cls = embedder_id.extra_value
    embedder = embedder_cls()
    doc_text = doc_obj["doc_text"]
    for chunk in doc_obj["chunks"]:
        start_index, end_index = chunk["doc_text_range"]
        chunk_text = doc_text[start_index:end_index]
        tokens = embedder.tokenize(chunk_text)
        embedding = embedder.embed(tokens)
        chunk["tokens"] = tokens
        chunk["embedding"] = embedding

    write_json(output_json_file, doc_obj)
