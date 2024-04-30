#!/usr/bin/env bash

set -e

EXEC_STR="$1"
DB_STR="$2"

if [[ "$EXEC_STR" =~ ^file=.+$ ]]; then
    EXEC_ARG_NAME=--file
    EXEC_ARG_VALUE="${EXEC_STR#file=}"
elif [[ "$EXEC_STR" =~ ^command=.+$ ]]; then
    EXEC_ARG_NAME=--command
    EXEC_ARG_VALUE="${EXEC_STR#command=}"
else
    echo "Invalid EXEC_STR: $EXEC_STR"
    exit 1
fi

if [[ "$DB_STR" =~ ^dbname=rag_v2_[a-zA-Z0-9_]+$ ]]; then
    DB_ARG="--dbname ${DB_STR#dbname=}"
elif [[ "$DB_STR" =~ ^dbset=rag_v2_[a-zA-Z0-9_]+$ ]]; then
    DB_ARG="--set RAG_DB_NAME=${DB_STR#dbset=}"
elif [[ -z "$DB_STR" ]]; then
    DB_ARG=""
else
    echo "Invalid DB_STR: $DB_STR"
    exit 1
fi

RAG_DB_HOST="postgresql-ebp.cfwmuvh4blso.us-west-2.rds.amazonaws.com"
RAG_DB_PORT=5432
RAG_DB_USERNAME="postgres"

PGPASSWORD="$RAG_DB_PASSWORD" psql \
    --host "$RAG_DB_HOST" \
    --port "$RAG_DB_PORT" \
    --username "$RAG_DB_USERNAME" \
    $EXEC_ARG_NAME "$EXEC_ARG_VALUE" \
    $DB_ARG
