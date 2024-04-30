import inspect
from typing import Callable

import uvicorn
from fastapi import FastAPI

# NOTE:
# The below is left over from an earlier design direction.
# Just commiting it for future re-working for newer design.

# TODO:
# * Support various embedders
# * Large batch doc inserts
# * User auth'n and auth'z
# * List paging
# * Entity filter query args
# * Attribute filter query args


def _get_frame_details(frame):
    args, _, _, locals = inspect.getargvalues(frame)
    return {"name": frame.f_code.co_name, "args": {arg: locals[arg] for arg in args}}


def _list_databases():
    return _get_frame_details(inspect.currentframe())


def _create_database():
    return _get_frame_details(inspect.currentframe())


def _get_database(database_id: str):
    return _get_frame_details(inspect.currentframe())


def _update_database(database_id: str):
    return _get_frame_details(inspect.currentframe())


def _delete_database(database_id: str):
    return _get_frame_details(inspect.currentframe())


def _list_documents(database_id: str):
    return _get_frame_details(inspect.currentframe())


def _add_document(database_id: str):
    return _get_frame_details(inspect.currentframe())


def _get_document(database_id: str, document_id: str):
    return _get_frame_details(inspect.currentframe())


def _delete_document(database_id: str, document_id: str):
    return _get_frame_details(inspect.currentframe())


def _get_document_content(database_id: str, document_id: str):
    return _get_frame_details(inspect.currentframe())


def _list_document_embeddings(database_id: str, document_id: str):
    return _get_frame_details(inspect.currentframe())


def _list_document_references(database_id: str, document_id: str):
    return _get_frame_details(inspect.currentframe())


def _list_document_uploads(database_id: str, document_id: str):
    return _get_frame_details(inspect.currentframe())


def _list_embeddings(database_id: str):
    return _get_frame_details(inspect.currentframe())


def _get_embedding(database_id: str, embedding_id: str):
    return _get_frame_details(inspect.currentframe())


def _list_embedding_documents(database_id: str, embedding_id: str):
    return _get_frame_details(inspect.currentframe())


def _list_embedding_references(database_id: str, embedding_id: str):
    return _get_frame_details(inspect.currentframe())


def _list_references(database_id: str):
    return _get_frame_details(inspect.currentframe())


def _get_reference(database_id: str, reference_id: str):
    return _get_frame_details(inspect.currentframe())


def _get_reference_document(database_id: str, reference_id: str):
    return _get_frame_details(inspect.currentframe())


def _get_reference_embedding(database_id: str, reference_id: str):
    return _get_frame_details(inspect.currentframe())


def _list_uploads(database_id: str):
    return _get_frame_details(inspect.currentframe())


def _get_upload(database_id: str, upload_id: str):
    return _get_frame_details(inspect.currentframe())


def _get_upload_document(database_id: str, upload_id: str):
    return _get_frame_details(inspect.currentframe())


_databases_routes_tree = {
    "databases": {
        "GET": _list_databases,
        "POST": _create_database,
        "{database_id}": {
            "GET": _get_database,
            "PATCH": _update_database,
            "DELETE": _delete_database,
            "documents": {
                "GET": _list_documents,
                "POST": _add_document,
                "{document_id}": {
                    "GET": _get_document,
                    "DELETE": _delete_document,
                    "content": {
                        "GET": _get_document_content,
                    },
                    "embeddings": {
                        "GET": _list_document_embeddings,
                    },
                    "references": {
                        "GET": _list_document_references,
                    },
                    "uploads": {
                        "GET": _list_document_uploads,
                    },
                },
            },
            "embeddings": {
                "GET": _list_embeddings,
                "{embedding_id}": {
                    "GET": _get_embedding,
                    "documents": {
                        "GET": _list_embedding_documents,
                    },
                    "references": {
                        "GET": _list_embedding_references,
                    },
                },
            },
            "references": {
                "GET": _list_references,
                "{reference_id}": {
                    "GET": _get_reference,
                    "document": {
                        "GET": _get_reference_document,
                    },
                    "embedding": {
                        "GET": _get_reference_embedding,
                    },
                },
            },
            "uploads": {
                "GET": _list_uploads,
                "{upload_id}": {
                    "GET": _get_upload,
                    "document": {
                        "GET": _get_upload_document,
                    },
                },
            },
        },
    }
}


def add_api_routes_helper(app: FastAPI, base_route: str, sub_routes_tree: dict):
    for key, value in sub_routes_tree.items():
        assert isinstance(key, str)
        if key in ["GET", "POST", "PATCH", "DELETE"]:
            assert isinstance(value, Callable)
            app.add_api_route(base_route, value, methods=[key])
        else:
            assert isinstance(value, dict)
            add_api_routes_helper(app, f"{base_route}/{key}", value)


def add_api_routes(app: FastAPI):
    add_api_routes_helper(app, "/v2/rag", _databases_routes_tree)


def main():
    app = FastAPI()
    add_api_routes(app)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
