# Steps for testing the RAG service.

All the steps below are relative to the TNE troopship repo.

```sh
# Set as appropriate:
MY_TNE_ROOT=~/work/repos/github.com/TNE-ai
```
# First setup/connect the local/k8s RAG service.

## Setting up local RAG service for testing.

```sh
# Run these commands for local RAG service testing.
cd $MY_TNE_ROOT/troopship/rag
docker build . -t rag:latest
docker run --rm -p 8080:8080 -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN rag:latest
```

You should see output like this:

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
Transient error StatusCode.UNAVAILABLE encountered while exporting traces to localhost:4317, retrying in 1s.
Transient error StatusCode.UNAVAILABLE encountered while exporting traces to localhost:4317, retrying in 2s.
Transient error StatusCode.UNAVAILABLE encountered while exporting traces to localhost:4317, retrying in 4s.
```

The transient errors above occur because OpenTelemetry isn't configured in this simple scenario. You can ignore these errors.

Now you can test the local RAG service at http://localhost:8080/v2/rag

See below for [running tests](#running-tests).

## Connecting to k8s RAG service for testing.

```sh
# Run this command for k8s RAG service testing.
kubectl port-forward service/rag-svc 8080:8080 -n agents
```

You should see output like this:

```
Forwarding from 127.0.0.1:8080 -> 8080
Forwarding from [::1]:8080 -> 8080
```

Now you can test the k8s RAG service at http://localhost:8080/v2/rag

See below for [running tests](#running-tests).

# Then run some tests.

## Test env setup.

First, setup for either local or k8s testing by following the instructions above.

Then, make sure you have a local (not committed) local_secrets.env file here:

```sh
cat $MY_TNE_ROOT/troopship/rag/v2/test/local_secrets.env 
```

Which should have env vars defined (with your values substituted for ...) like so:

```
EMBEDDINGS_DB_PASSWORD=...
OPENAI_API_KEY=...
```

Then, setup the local dev environment (necessary for the python test scripts below):

```sh
cd $MY_TNE_ROOT/troopship/rag
python -m venv .rag-venv  # If not already created.
source .rag-venv/bin/activate  # If not already activated.
pip install -r requirements_dev.txt  # If not already installed.
```

## To test the old v1 API:

```sh
cd $MY_TNE_ROOT/troopship/rag/v1/tests
make test_request_0_history
```

After roughly 30 seconds (v1 is not streaming) you should see a json response object output.

## To test the new v2 API:

```sh
cd $MY_TNE_ROOT/troopship/rag
./v2/test/rag_test_1.sh
```

You should see a bunch of lines output that begin like this:

```
{"thread_id":"67c68961-f091-4db7-a614-4014743f9957","patch_record":{...
{"thread_id":"67c68961-f091-4db7-a614-4014743f9957","patch_record":{...
{"thread_id":"67c68961-f091-4db7-a614-4014743f9957","patch_record":{...
```

Those are the streamed patch records which can be merged into the original request record, for a complete record of the particular request/response iteration.

## To test the new v2 API a bit more:

Next use this python test script, which does roughly the same thing as the above shell script, but also saves the received patch records to the given file:

```sh
python -m v2.test.simple_test_client <(v2/test/rag_test_1_json.sh) /tmp/patch_records.txt
less /tmp/patch_records.txt
```

You can load the saved patch records and merge them all together into a complete record, which is then printed, using this command:

```sh
python -m v2.test.print_merged_patch_records /tmp/patch_records.txt
```

If you want to interactively visualize the patch records, try this command:

```sh
python -m v2.test.playback_patch_records /tmp/patch_records.txt
```

To control playback:
* Right arrow applies the next patch record.
* Left arrow reverts the last applied patch record.
* Down arrow scrolls down if text overflows below.
* Up arrow scrolls up if the text overflows above.
* Any other key quits.

## To test v2 interactively:

Both `v2/test/rag_test_1.sh` and `v2/test/simple_test_client.py` simply send a single canned request and output the response in one way or another. There is now a test client that is an interactive loop allowing a sequence of multiple user inputs, and maintains (transient) history of the request/response thread. Here's how you run it:

```sh
# Setup of venv and local_secrets.env described above.
cd $MY_TNE_ROOT/troopship/rag
python -m v2.test.test_client v2/test/local_secrets.env
```
