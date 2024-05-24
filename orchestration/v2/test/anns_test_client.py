import sys
import time

import requests

from v2.api.util import (
    anns_request_from_json_str,
    anns_request_to_json_str,
    anns_response_from_json_str,
    anns_response_to_yaml_str,
)

_service_url = "http://localhost:8080/v2/anns"


def main():
    _, request_file = sys.argv

    with open(request_file) as request_fp:
        request = anns_request_from_json_str(request_fp.read())
    request_bytes = anns_request_to_json_str(request).encode()

    start_seconds = time.time()
    response_obj = requests.post(
        _service_url, data=request_bytes, headers={"Content-Type": "application/json"}
    )
    elapsed_seconds = time.time() - start_seconds
    status_code = response_obj.status_code

    if status_code == 200:
        response = anns_response_from_json_str(response_obj.content.decode())
        response = anns_response_to_yaml_str(response)
    else:
        response = None

    print(f"elapsed_seconds:  {elapsed_seconds}")
    print(f"status_code:      {status_code}")
    print(response)


if __name__ == "__main__":
    main()
