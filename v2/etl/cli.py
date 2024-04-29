from v2.etl.cmd_cleanse_combine_and_split import cleanse_combine_and_split
from v2.etl.cmd_extract_pdf_text import extract_pdf_text
from v2.etl.cmd_insert_embeddings_and_sources import insert_embeddings_and_sources
from v2.etl.cmd_tokenize_and_embed import tokenize_and_embed
from v2.etl.typer_util import typer_cli_with_commands

etl_cli = typer_cli_with_commands(
    "etl",
    "Extract Transform & Load (ETL) commands.",
    [
        cleanse_combine_and_split,
        extract_pdf_text,
        insert_embeddings_and_sources,
        tokenize_and_embed,
    ],
)

if __name__ == "__main__":
    etl_cli()
