import os

import sys, copy
sys.path.append('../')
import myprompt

from transformers import FuyuProcessor, FuyuForCausalLM
import torch
from PIL import Image
import random, json, time
from tqdm import tqdm


MODEL_NAME = 'fuyu-8b'
NOW_ROOT = myprompt.now_root
SAVE_ROOT = f'Eval/{myprompt.task_name}'
CKPT_PATH = f'/data/FinAi_Mapping_Knowledge/qiyiyan/models/{MODEL_NAME}'


def scale_image(image, max_size=1080):
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_width = int(width * max_size / height)
        new_height = max_size
    scaled_image = image.resize((new_width, new_height), Image.BICUBIC)
    return scaled_image


def query_once(processor, model, raw_image, question):
    Q_base = copy.deepcopy(myprompt.prompt_yes_or_no)
    Q_base = Q_base.format(question)
    max_new_tokens = 20
    inputs = processor(text=Q_base, images=raw_image, return_tensors="pt")
    for k, v in inputs.items(): 
        inputs[k] = v.to("cuda")
    inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape, device="cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=model.config.eos_token_id)
    # out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = processor.batch_decode(out[:, -max_new_tokens:], skip_special_tokens=True)
    return Q_base, answer


def query():
    model = FuyuForCausalLM.from_pretrained(CKPT_PATH, device_map="cuda", torch_dtype=torch.float16)
    processor = FuyuProcessor.from_pretrained(CKPT_PATH)
    
    QA_meta_list = myprompt.load_meta()
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
                raw_image = scale_image(raw_image)
                for key in QAs.keys():
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
