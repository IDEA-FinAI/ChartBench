import os, re, json
import numpy as np

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


def summary_inference(model_name, task_name):

    print(model_name, '\t', task_name)
    SAVE_ROOT = f'Eval/{task_name}'
    QA_meta_list = load_meta()
    inf_time_all = []
    for QA_path in QA_meta_list:
        QA_path = QA_path.replace('QA', SAVE_ROOT)
        QA_path = QA_path.replace('meta.json', f'{model_name}.json')
        # print(QA_path) # For debug
        with open(QA_path, 'r', encoding='utf-8') as fj:
            meta = json.load(fj)
            file_list = list(meta.keys())
            for file in file_list:
                inf_time_file = float(meta[file]["InfTime"])
                inf_time_all.append(inf_time_file)
                
    inf_time_avg = np.mean(inf_time_file) / 8 
    print(inf_time_avg, '\n')


if __name__ == '__main__':
    
    model_names = [
        'blip2-flan-t5-xxl',
        'cogvlm-chat', 
        # 'fuyu-8b',
        'instructblip-vicuna-7b',
        'internlm-xcomposer-7b',
        #  'LaVIN',
        'llava-v1.5-13b',
        'minigpt_v2',
        'mplug-owl-bloomz-7b-multilingual',
        'Qwen-VL-Chat',
        'shikra-7b',
        'SPHINX',
        'visualglm-6b',
        'ChartLlama-13b'
        ]

    tasks = [
        # 'prompt_no_or_yes', 
        # 'prompt_yes_or_no', 
        # 'prompt_yes_or_no_blip', 
        'BLIP2_Style'
        ]


    for task in tasks:
        for model_name in model_names:
            summary_inference(model_name, task)