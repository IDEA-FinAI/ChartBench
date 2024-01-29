import os

import sys, copy
sys.path.append('../')
import myprompt

from PIL import Image
import random, json, time
from tqdm import tqdm

import sys
import logging
import time
import argparse
import tempfile
from pathlib import Path
from typing import List, Any, Union
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
from mmengine import Config
import transformers
from transformers import BitsAndBytesConfig

# sys.path.append(str(Path(__file__).parent.parent.parent))

from mllm.dataset.process_function import PlainBoxFormatter
from mllm.dataset.builder import prepare_interactive
from mllm.utils import draw_bounding_boxes
from mllm.models.builder.build_shikra import load_pretrained_shikra


MODEL_NAME = 'shikra-7b'
NOW_ROOT = myprompt.now_root
SAVE_ROOT = f'Eval/{myprompt.task_name}'
CKPT_PATH = f'/data/FinAi_Mapping_Knowledge/qiyiyan/models/shikra-7b'

log_level = logging.WARNING
transformers.logging.set_verbosity(log_level)
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

TEMP_FILE_DIR = Path(__file__).parent / 'temp'
TEMP_FILE_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser("Shikra Web Demo")
parser.add_argument('--load_in_8bit', action='store_true')
parser.add_argument('--server_name', default=None)
parser.add_argument('--server_port', type=int, default=None)

args = parser.parse_args()

model_args = Config(dict(
    type='shikra',
    version='v1',

    # checkpoint config
    cache_dir=None,
    model_name_or_path=CKPT_PATH,
    vision_tower=r'openai/clip-vit-large-patch14',
    pretrain_mm_mlp_adapter=None,

    # model config
    mm_vision_select_layer=-2,
    model_max_length=2048,

    # finetune config
    freeze_backbone=False,
    tune_mm_mlp_adapter=False,
    freeze_mm_mlp_adapter=False,

    # data process config
    is_multimodal=True,
    sep_image_conv_front=False,
    image_token_len=256,
    mm_use_im_start_end=True,

    target_processor=dict(
        boxes=dict(type='PlainBoxFormatter'),
    ),

    process_func_args=dict(
        conv=dict(type='ShikraConvProcess'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='ShikraTextProcess'),
        image=dict(type='ShikraImageProcessor'),
    ),

    conv_args=dict(
        conv_template='vicuna_v1.1',
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=None),
    ),

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
))

training_args = Config(dict(
    bf16=False,
    fp16=True,
    device='cuda',
    fsdp=None,
))

if args.load_in_8bit:
    quantization_kwargs = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
        )
    )
else:
    quantization_kwargs = dict()


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    

def query_once(preprocessor, tokenizer, model, image_path, question):
    
    do_sample = False
    max_length = 2048
    ds = prepare_interactive(model_args, preprocessor)

    Q_base = copy.deepcopy(myprompt.prompt_yes_or_no)
    Q_base = Q_base.format(question)

    image = Image.open(image_path).convert("RGB")
    image = expand2square(image)
    ds.set_image(image)
    ds.append_message(role=ds.roles[0], message=Q_base, boxes=[], boxes_seq=[])

    model_inputs = ds.to_model_input()
    model_inputs['images'] = model_inputs['images'].to(torch.float16)

    gen_kwargs = dict(
        use_cache=True,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_length,
    )
        
    input_ids = model_inputs['input_ids']
    with torch.inference_mode():
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            output_ids = model.generate(**model_inputs, **gen_kwargs)
    input_token_len = input_ids.shape[-1]
    response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

    return Q_base, response


def query():

    model, preprocessor = load_pretrained_shikra(model_args, training_args, **quantization_kwargs)
    if not getattr(model, 'is_quantized', False):
        model.to(dtype=torch.float16, device=torch.device('cuda'))
    if not getattr(model.model.vision_tower[0], 'is_quantized', False):
        model.model.vision_tower[0].to(dtype=torch.float16, device=torch.device('cuda'))
    print(f"LLM device: {model.device}, is_quantized: {getattr(model, 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
    print(f"vision device: {model.model.vision_tower[0].device}, is_quantized: {getattr(model.model.vision_tower[0], 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")

    preprocessor['target'] = {'boxes': PlainBoxFormatter()}
    tokenizer = preprocessor['text']

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
                # raw_image = Image.open(image_dir).convert('RGB')
                for key in QAs.keys():
                    logger.write(image_dir + '\t' + key + '\n')
                    Qr = meta[file]["QA"][key]['Qr']
                    Qw = meta[file]["QA"][key]['Qw']
                    DIY_Qr, DIY_Ar = query_once(preprocessor, tokenizer, model, image_dir, Qr)
                    DIY_Qw, DIY_Aw = query_once(preprocessor, tokenizer, model, image_dir, Qw)
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