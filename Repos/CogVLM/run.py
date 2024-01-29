import os

import sys, copy
sys.path.append('../')
import myprompt

import torch
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor
from utils.chat import chat
from sat.model.mixins import CachedAutoregressiveMixin
import argparse
# import bitsandbytes
from PIL import Image
import random, json, time
from tqdm import tqdm


MODEL_NAME = 'cogvlm-chat'
SAVE_ROOT = f'Eval/{myprompt.task_name}'
NOW_ROOT = myprompt.now_root
CKPT_PATH = f'/data/FinAi_Mapping_Knowledge/qiyiyan/models/CogVLM/{MODEL_NAME}'
TOKENIZER_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/models/vicuna-7b-v1.5'


def query_once(text_processor_infer, image_processor, model, raw_image, question):
    Q_base = copy.deepcopy(myprompt.prompt_yes_or_no)
    Q_base = Q_base.format(question)
    with torch.no_grad():
        answer, _, _ = chat(
            raw_image, 
            model,
            text_processor_infer,
            image_processor,
            Q_base, 
            history=[],
            max_length=2048, 
            top_p=0.4, 
            temperature=0.8,
            top_k=1,
            invalid_slices=text_processor_infer.invalid_slices,
            no_prompt=False
            )
    return Q_base, answer


def query():
    # load model
    model, model_args = CogVLMModel.from_pretrained(
        CKPT_PATH,
        args=argparse.Namespace(
            deepspeed=None,
            local_rank=0,
            rank=0,
            world_size=1,
            model_parallel_size=1,
            mode='inference',
            skip_init=True,
            fp16=False,
            bf16=True,
            use_gpu_initialization=True,
            device='cuda',
        ))
    model = model.eval()

    tokenizer = llama2_tokenizer(TOKENIZER_PATH, signal_type="chat")
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    text_processor_infer = llama2_text_processor_inference(tokenizer, None, model.image_length)

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
                for key in QAs.keys():
                    Qr = meta[file]["QA"][key]['Qr']
                    Qw = meta[file]["QA"][key]['Qw']
                    DIY_Qr, DIY_Ar = query_once(text_processor_infer, image_processor, model, image_dir, Qr)
                    DIY_Qw, DIY_Aw = query_once(text_processor_infer, image_processor, model, image_dir, Qw)
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









