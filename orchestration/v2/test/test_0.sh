#!/usr/bin/env bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

"$SCRIPT_DIR/test_0_json.sh" \
	| curl http://localhost:8080/v2/rag -H "Content-Type: application/json" -d @-
