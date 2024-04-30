# RAG v2 ETL scripts.

These are temporary scripts for extracting, transforming, and loading text and embeddings from PDFs into a databases for RAG v2. This is a very manual CLI process because these scripts are adapted from v1 (which only ever had a single DB). This is for tempoarary use until this CLI process is replaced by a proper ETL GUI.

## Setting up.

First create a RAG embeddings database for you particular document set by following [these DB admin instrustions](../db_admin/README.md).

```sh
cd your/tne/root/troopship/rag
cat v2/etl/db_config.env
```

In `db_config.env` replace `foobar` with the name of your document DB in this line:

```sh
POSTGRES_DB=rag_v2_foobar
```

## Performanig ETL.

TODO(Guy): Write-up this process.

```sh
~/your_tne_root/troopship/rag$ python -m v2.etl.cli --help
                                                                                                                          
 Usage: python -m v2.etl.cli [OPTIONS] COMMAND [ARGS]...                                                                  
                                                                                                                          
 Extract Transform & Load (ETL) commands.                                                                                 
                                                                                                                          
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.         │
│ --help                        Show this message and exit.                                                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ cleanse-combine-and-split                                              TODO: Add help text.                            │
│ extract-pdf-text                                                       TODO: Add help text.                            │
│ insert-embeddings-and-sources                                          TODO: Add help text.                            │
│ tokenize-and-embed                                                     TODO: Add help text.                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```sh
~/your_tne_root/troopship/rag$ python -m v2.etl.cli extract-pdf-text --help
                                                                                                                          
 Usage: python -m v2.etl.cli extract-pdf-text [OPTIONS] INPUT_PDF_FILE                                                    
                                              OUTPUT_JSON_FILE                                                            
                                                                                                                          
 TODO: Add help text.                                                                                                     
                                                                                                                          
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    input_pdf_file        TEXT  TODO: Add help text. [default: None] [required]                                       │
│ *    output_json_file      TEXT  TODO: Add help text. [default: None] [required]                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --pdf-extracter-id        [pdfminer|pdfplumber|pymupdf|pypdf|pypdfium2]  TODO: Add help text. [default: pypdfium2]     │
│ --help                                                                   Show this message and exit.                   │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```sh
~/your_tne_root/troopship/rag$ python -m v2.etl.cli cleanse-combine-and-split --help
                                                                                                                          
 Usage: python -m v2.etl.cli cleanse-combine-and-split [OPTIONS]                                                          
                                                       INPUT_JSON_FILE                                                    
                                                       OUTPUT_JSON_FILE                                                   
                                                                                                                          
 TODO: Add help text.                                                                                                     
                                                                                                                          
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    input_json_file       TEXT  TODO: Add help text. [default: None] [required]                                       │
│ *    output_json_file      TEXT  TODO: Add help text. [default: None] [required]                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --text-splitter-id            [char|nltk|spacy]  TODO: Add help text. [default: char]                                  │
│ --target-chunk-size           INTEGER            TODO: Add help text. [default: 1000]                                  │
│ --target-chunk-overlap        INTEGER            TODO: Add help text. [default: 500]                                   │
│ --help                                           Show this message and exit.                                           │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```sh
~/your_tne_root/troopship/rag$ python -m v2.etl.cli tokenize-and-embed --help
                                                                                                                          
 Usage: python -m v2.etl.cli tokenize-and-embed [OPTIONS] INPUT_JSON_FILE                                                 
                                                OUTPUT_JSON_FILE                                                          
                                                                                                                          
 TODO: Add help text.                                                                                                     
                                                                                                                          
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    input_json_file       TEXT  TODO: Add help text. [default: None] [required]                                       │
│ *    output_json_file      TEXT  TODO: Add help text. [default: None] [required]                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --embedder-id        [openai_ada_002]  TODO: Add help text. [default: openai_ada_002]                                  │
│ --help                                 Show this message and exit.                                                     │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```sh
~/your_tne_root/troopship/rag$ python -m v2.etl.cli insert-embeddings-and-sources --help
                                                                                                                          
 Usage: python -m v2.etl.cli insert-embeddings-and-sources [OPTIONS]                                                      
                                                           DB_CONFIG_FILE INPUT                                           
                                                           _JSON_FILES_FILE                                               
                                                                                                                          
 TODO: Add help text.                                                                                                     
                                                                                                                          
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    db_config_file             TEXT  TODO: Add help text. [default: None] [required]                                  │
│ *    input_json_files_file      TEXT  TODO: Add help text. [default: None] [required]                                  │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --batch-size                                         INTEGER  TODO: Add help text. [default: 1000]                     │
│ --validate-embeddings    --no-validate-embeddings             TODO: Add help text. [default: validate-embeddings]      │
│ --help                                                        Show this message and exit.                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
