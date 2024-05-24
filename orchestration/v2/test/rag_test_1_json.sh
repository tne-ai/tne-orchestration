#!/usr/bin/env bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source "$SCRIPT_DIR/local_secrets.env"
export EMBEDDINGS_DB_PASSWORD
export OPENAI_API_KEY
envsubst '$EMBEDDINGS_DB_PASSWORD $OPENAI_API_KEY' < "$SCRIPT_DIR/rag_test_1.json"
