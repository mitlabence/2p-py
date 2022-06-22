import json

def json_write(json_fpath: str, json_dict: dict):
    with open(json_fpath, "w") as f:
        json.dump(json_dict, f, indent=4)

def json_read(json_fpath: str):
    with open(json_fpath, 'r') as json_file:
        json_dict = json.load(json_file)
    return json_dict
