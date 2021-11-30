import json
import os
import re

def json_load(json_path):
    with open(json_path, encoding='UTF-8') as json_file:
        json_data = json.load(json_file)
    return json_data

test_dir = '../runs/test'
test_folders = os.listdir(test_dir)
test_folders = [re.sub('exp','',name) for name in test_folders]
print(test_folders)
