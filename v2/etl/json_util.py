import json


def write_json(file, obj):
    with open(file, "w") as fp:
        json.dump(obj, fp, indent=4)


def read_json(file):
    with open(file) as fp:
        return json.load(fp)
