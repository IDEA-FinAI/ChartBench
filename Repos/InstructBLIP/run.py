import os

import sys, copy
sys.path.append('../')
import myprompt

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import random, json, time
from tqdm import tqdm


MODEL_NAME = 'instructblip-vicuna-7b'
NOW_ROOT = myprompt.now_root
SAVE_ROOT = f'Eval/{myprompt.task_name}'
CKPT_PATH = f'/data/FinAi_Mapping_Knowledge/qiyiyan/models/{MODEL_NAME}'


def query_once(processor, model, raw_image, question):
    Q_base = copy.deepcopy(myprompt.prompt_yes_or_no)
    Q_base = Q_base.format(question)
    inputs = processor(images=raw_image, text=Q_base, return_tensors="pt").to('cuda')
    outputs = model.generate(
            **inputs,
            do_sample=True,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
    )
    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return Q_base, answer


def query():
    model = InstructBlipForConditionalGeneration.from_pretrained(CKPT_PATH)
    processor = InstructBlipProcessor.from_pretrained(CKPT_PATH)
    model.to('cuda')
    
    QA_meta_list = myprompt.load_meta()
    logger = open('./log.txt', 'w')
    file_idx = 1
    for QA_path in QA_meta_list:
        print(f'No. {file_idx}: ' + QA_path)
        file_idx += 1
        answer_path = QA_path.replace('QA', SAVE_ROOT)
        answer_path = answer_path.replace('meta.json', '')
        os.makedirs(answer_path, exist_ok=True)
        answer_path = os.path.join(answer_path, f'{MODEL_NAME}.json')
        # if os.path.exists(answer_path): continue
        
        with open(QA_path, 'r') as fmeta:
            meta = json.load(fmeta)
            file_list = list(meta.keys())
            for file in tqdm(file_list):
                # if file == '21.txt': continue
                start_time = time.time()
                QAs = meta[file]["QA"]
                image_dir = meta[file]['image_path']
                image_dir = os.path.join(NOW_ROOT, image_dir)
                raw_image = Image.open(image_dir).convert('RGB')
                for key in QAs.keys():
                    logger.write(image_dir + '\t' + key + '\n')
                    Qr = meta[file]["QA"][key]['Qr']
                    Qw = meta[file]["QA"][key]['Qw']
                    DIY_Qr, DIY_Ar = query_once(processor, model, raw_image, Qr)
                    DIY_Qw, DIY_Aw = query_once(processor, model, raw_image, Qw)
                    meta[file]["QA"][key]['Qr'] = DIY_Qr
                    meta[file]["QA"][key]['Ar'] = DIY_Ar
                    meta[file]["QA"][key]['Qw'] = DIY_Qw
                    meta[file]["QA"][key]['Aw'] = DIY_Aw
                end_time = time.time()
                run_time = end_time - start_time
                meta[file]["InfTime"] = str(run_time)
                
        with open(answer_path, 'w', encoding='utf-8') as fj:
            fj.write(json.dumps(meta, indent=4, ensure_ascii=False))
        # exit()


if __name__ == "__main__":
    query()