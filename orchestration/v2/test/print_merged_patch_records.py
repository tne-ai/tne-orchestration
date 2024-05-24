import sys

from v2.api.api import RagRecord
from v2.api.util import merge_records, rag_record_to_yaml_str
from v2.test.simple_test_client import read_patch_records


def main():
    _, patch_records_file = sys.argv
    patch_records = read_patch_records(patch_records_file)
    merged_record = RagRecord()
    for patch_record in patch_records:
        merged_record = merge_records(merged_record, patch_record)
    print(rag_record_to_yaml_str(merged_record, "merged_record"))


if __name__ == "__main__":
    main()
