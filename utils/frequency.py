import os
import re
import pandas as pd
import yaml

class getFrequency:
    def __init__(self, JSON_DATA):
        self.JSON_DATA = JSON_DATA
        self.DIR = JSON_DATA['DIR']['ROOT_DIR']
        self.TRAIN_TXT_PATH = os.path.join(self.DIR, 'train.txt')
        self.VALID_TXT_PATH = os.path.join(self.DIR, 'valid.txt')
        self.LABELS_FOLDER_PATH = os.path.join(self.DIR, 'labels')
        self.CLASSES_NAMES = os.path.join(self.DIR, 'classes.names')
        self.DATA_YAML = os.path.join(self.DIR, 'data.yaml')
        f = open(self.CLASSES_NAMES, 'r')
        self.LABELS = f.readlines()
        self.LABELS = [re.sub('\n','',label) for label in self.LABELS]
        self.make_yaml()

    def make_yaml(self):
        data = {}
        data['train'] = re.sub('[\\\]','/',self.TRAIN_TXT_PATH)
        data['val'] = re.sub('[\\\]','/',self.VALID_TXT_PATH)
        data['nc'] = len(self.LABELS)
        data['names'] = self.LABELS
        with open(self.DATA_YAML, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True)

    def extract_class_id(self, txt_path):
        class_ids = []
        txt = open(txt_path, 'r')
        lines = txt.readlines()
        for line in lines:
            class_ids.append(line.split(' ')[0])
        return class_ids

    def getValidTxt(self):
        txt = open(self.VALID_TXT_PATH)
        lines = txt.readlines()
        paths = []
        for line in lines:
            line = re.sub('[\\\n]','',line)
            line = line.split('.')[0]
            line = line.split('/')
            paths.append(f"{line[len(line)-1]}.txt")
        return paths

    def getClassFreq(self, is_valid=False):
        freq_data = {}
        for i in range(len(self.LABELS)):
            freq_data[str(i)] = 0

        if is_valid:
            save_path = './TEST_DATA_FREQUENCY.csv'
            label_txt_list = self.getValidTxt()
        else:
            save_path = './TOTAL_DATA_FREQUENCY.csv'
            label_txt_list = os.listdir(self.LABELS_FOLDER_PATH)

        for txt_name in label_txt_list:
            txt_path = os.path.join(self.LABELS_FOLDER_PATH, txt_name)
            class_ids = self.extract_class_id(txt_path)
            for class_id in class_ids:
                freq_data[class_id] = freq_data[class_id] + 1

        f_class = list(range(len(self.LABELS)))
        f_freq = []
        for i in f_class:
            f_freq.append(freq_data[str(i)])
        df = pd.DataFrame({'Class ID':f_class, 'Frequency':f_freq}, columns=['Class ID', 'Frequency'])

        df.to_csv(save_path, index=False, encoding='euc-kr')





