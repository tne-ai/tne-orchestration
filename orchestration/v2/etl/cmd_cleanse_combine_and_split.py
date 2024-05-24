# type: ignore

import re

from v2.etl.json_util import read_json, write_json
from v2.etl.typer_util import TyperEnum, typer_arg, typer_opt


def _cleanse_text(text: str):
    text = text.replace("\r\n", "\n")
    text = re.sub("[\x00-\x04\v\f\r]", "\n", text)
    text = re.sub("[\x00-\t\v-\x1f]", " ", text)
    text = re.sub(" {2,}", " ", text)
    text = re.sub(" \n", "\n", text)
    text = re.sub("\n ", "\n", text)
    text = re.sub("\n{3,}", "\n\n", text)

    # This character shows up particularly with pypdfium2
    # and is used as a mid-word line break.
    text = text.replace(f"{chr(65_534)}", "")

    text = text.lstrip(" \n")
    text = text.rstrip(" \n")
    text += "\n\n"

    return text


def _split_text(text, text_splitter_id, chunk_size, chunk_overlap):
    text_splitter_creator = text_splitter_id.extra_value
    text_splitter = text_splitter_creator(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=True,
        add_start_index=True,
    )
    docs = text_splitter.create_documents([text])
    chunk_text_and_start_index_pairs = [
        (doc.page_content, doc.metadata["start_index"]) for doc in docs
    ]
    return chunk_text_and_start_index_pairs


def _lazy_char_text_splitter_creator(*args, **kwargs):
    # Lazy import removes noticeable latency from showing --help.
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(*args, **kwargs)


def _lazy_nltk_text_splitter_creator(*args, **kwargs):
    # Lazy import removes noticeable latency from showing --help.
    from langchain.text_splitter import NLTKTextSplitter

    return NLTKTextSplitter(*args, **kwargs)


def _lazy_spacy_text_splitter_creator(*args, **kwargs):
    # Lazy import removes noticeable latency from showing --help.
    from langchain.text_splitter import SpacyTextSplitter

    return SpacyTextSplitter(*args, **kwargs)


class _TextSplitterId(TyperEnum):
    char = "char", _lazy_char_text_splitter_creator
    nltk = "nltk", _lazy_nltk_text_splitter_creator
    spacy = "spacy", _lazy_spacy_text_splitter_creator


def cleanse_combine_and_split(
    input_json_file: typer_arg(str, "TODO: Add help text."),
    output_json_file: typer_arg(str, "TODO: Add help text."),
    text_splitter_id: typer_opt(
        _TextSplitterId, "TODO: Add help text."
    ) = _TextSplitterId.char,
    target_chunk_size: typer_opt(int, "TODO: Add help text.") = 1_000,
    target_chunk_overlap: typer_opt(int, "TODO: Add help text.") = 500,
):
    """TODO: Add help text."""

    doc_obj = read_json(input_json_file)

    doc_obj["command_args"]["cleanse_combine_and_split_args"] = {
        "input_json_file": input_json_file,
        "output_json_file": output_json_file,
        "text_splitter_id": text_splitter_id.name,
        "target_chunk_size": target_chunk_size,
        "target_chunk_overlap": target_chunk_overlap,
    }

    pages = doc_obj["pages"]
    raw_extracted_texts = [page["raw_extracted_text"] for page in pages]
    cleansed_texts = [
        _cleanse_text(raw_extracted_text) for raw_extracted_text in raw_extracted_texts
    ]

    doc_text = ""
    for page, cleansed_text in zip(pages, cleansed_texts):
        doc_text_start_index = len(doc_text)
        doc_text_end_index = doc_text_start_index + len(cleansed_text)
        doc_text_range = [doc_text_start_index, doc_text_end_index]
        page["doc_text_range"] = doc_text_range
        doc_text += cleansed_text
    doc_obj["doc_text"] = doc_text

    chunk_text_and_start_index_pairs = _split_text(
        doc_text, text_splitter_id, target_chunk_size, target_chunk_overlap
    )
    chunks = []
    for chunk_text, doc_text_start_index in chunk_text_and_start_index_pairs:
        doc_text_end_index = doc_text_start_index + len(chunk_text)
        assert doc_text[doc_text_start_index:doc_text_end_index] == chunk_text
        doc_text_range = [doc_text_start_index, doc_text_end_index]
        chunks.append({"doc_text_range": doc_text_range})
    doc_obj["chunks"] = chunks

    write_json(output_json_file, doc_obj)
