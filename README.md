# TNE Orchestration

Legacy implementation of TNE LLM orchestration module. Used by `bp-runner` server.

## S3 Directory Structure

Currently, S3 directory structure is as follows:

```
bp-authoring-files/
├── d/ 
│ ├── <UID 1> 
│ │ ├── data/  # Miscellanous files, like CSV, text, images, etc.
│ │ ├── items/ # Insight Edge items (NOTE: these will soon be moved elsewhere
│ │ ├── manifests/   # SlashGPT manifests collection
│ │ ├── modules/  # Python code collection
│ │ ├── proc/  # Expert graph collection
│ ├── <UID 2> 
│ │ ├── data/
│ │ ├── items/
│ │ ├── manifests/ 
│ │ ├── modules/ 
│ │ ├── proc/ 
...
