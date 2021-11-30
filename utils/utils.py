import json
import os

def json_load(json_path):
    with open(json_path, encoding='UTF-8') as json_file:
        json_data = json.load(json_file)
    return json_data

def json_save(json_path, data):
    f = open(json_path, 'w', encoding='UTF-8')
    json.dump(data, f, ensure_ascii=False)

def convertToConerPoint(raw_input_dir, save_dir):
  json_list = os.listdir(raw_input_dir)
  os.makedirs(save_dir, exist_ok=True)
  for json_filename in json_list:
    path = os.path.join(raw_input_dir, json_filename)
    with open(path, 'r', encoding='utf-8') as json_file:
      data = json.load(json_file)
    keys = list(data.keys())
    for key in keys:
      anns = data[key]['annotations']['Bbox Annotation']['Box']
      save_path = os.path.join(save_dir, f'{key}.txt')
      data_array = []
      data_array.append(len(anns))
      txt_result = open(save_path, 'w')
      for i, ann in enumerate(anns):
        category = ann['category_id']
        xmin = ann['x'] - ann['w']
        ymin = ann['y']
        xmax = ann['x']
        ymax = ann['y'] + ann['h']
        if i != len(anns)-1:
          txt_result.write(f'{category} {int(xmin)} {int(ymin)} {int(xmax)} {int(ymax)}\n')
        else:
          txt_result.write(f'{category} {int(xmin)} {int(ymin)} {int(xmax)} {int(ymax)}')
      txt_result.close()