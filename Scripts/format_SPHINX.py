import os, re, json
import pandas as pd

# env = xzz_2.0

META_PATH = './QA/Acc+/index.json'


def load_meta():
    QA_meta_list = []
    with open(META_PATH, 'r') as fmeta:
        meta = json.load(fmeta)
        chart_type = list(meta.keys())
        for chart in chart_type:
            for image_type in meta[chart].keys():
                QA_path = meta[chart][image_type]['QA_path']
                QA_meta_list.append(QA_path)
    return QA_meta_list


def format(task_name):
    SAVE_ROOT = f'Eval/{task_name}'
    QA_meta_list = load_meta()
    for QA_path in QA_meta_list:
        QA_path = QA_path.replace('QA', SAVE_ROOT)
        QA_path = QA_path.replace('meta.json', f'SPHINX.json')
        # print(QA_path) # For debug
        with open(QA_path, 'r', encoding='utf-8') as fj:
            lines = fj.readlines()
            lines[2001] = '}'
            lines = lines[:2002]
        with open(QA_path, 'w', encoding='utf-8') as fj:
            for line in lines:
                fj.write(line)


if __name__ == '__main__':
    
    tasks = [
        # 'prompt_no_or_yes', 
        # 'prompt_yes_or_no', 
        # 'prompt_yes_or_no_blip', 
        # 'prompt_yes_or_no_example'
        'prompt_cot_v1',
        ]

    for task in tasks:
        format(task)