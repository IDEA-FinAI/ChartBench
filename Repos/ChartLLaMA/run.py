import os

import sys, copy
sys.path.append('../')
import myprompt

import random, json, time
from PIL import Image

import argparse
import torch
import json
from tqdm import tqdm
import shortuuid
import warnings
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
from llava.model import *
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

import math


MODEL_NAME = 'ChartLlama-13b'
SAVE_ROOT = f'Eval/{myprompt.task_name}'
NOW_ROOT = myprompt.now_root
CKPT_PATH = f'/data/FinAi_Mapping_Knowledge/qiyiyan/models/{MODEL_NAME}'


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    # Load LLaVA model
    if model_base is None:
        raise ValueError('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
    if model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading LLaVA from base model...')
        model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional LLaVA weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')
            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')

    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["conversations"][0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        # print("====================")
        # print(qs)

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print("====================")
        # print(prompt)
        # exit()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def query_once(args, tokenizer, image_processor, model, model_name):

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = json.load(open(os.path.expanduser(args.question_file), 'r'))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["id"]
        cur_prompt = line["conversations"][0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()

        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1636,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "question": cur_prompt,
                                   "answer": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


def build_question_image_json(QA_path):

    Q_base = copy.deepcopy(myprompt.prompt_yes_or_no)
    pre_root = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/QA/'
    qet_root = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/Repo/ChartLLaMA/Cache/'
    base_root = QA_path.replace(pre_root, qet_root).replace('meta.json', '')
    os.makedirs(base_root, exist_ok=True)
    question_path_for_chartllama = base_root + 'question.json'
    answer_path_for_chartllama = base_root + 'answer.jsonl'
    
    idx = 1
    all_QA = []
    with open(QA_path, 'r') as fmeta:
        meta = json.load(fmeta)
        file_list = list(meta.keys())
        for file in file_list:
            image_dir = meta[file]['image_path']
            image_dir = os.path.join(NOW_ROOT, image_dir)
            QAs = meta[file]["QA"]
            for key in QAs.keys():
                for quest in ['Qr', 'Qw']:
                    Q = Q_base.format(meta[file]["QA"][key][quest])
                    item = {
                        "id": idx,
                        "image": image_dir,
                        "conversations": [
                            {"value": Q}
                        ],
                        "chart_type": meta[file]["chart_type"],
                        "image_type": meta[file]["image_type"],
                        "QA_type": meta[file]["QA_type"],
                        "file_name": file,
                        "YesorNo": quest,
                        "QA_key": key
                    }
                    idx += 1
                    all_QA.append(item)

    with open(question_path_for_chartllama, 'w') as file:
        json.dump(all_QA, file, indent=4, ensure_ascii=False)

    return question_path_for_chartllama, answer_path_for_chartllama, meta


def prase_answer_json(Q_chart_path, A_chart_path, meta):

    with open(Q_chart_path, 'r') as file:
        quest_meta = json.load(file)
        
    with open(A_chart_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            Q_ID = json_obj["question_id"]
            Q_line = json_obj["question"]
            A_line = json_obj["answer"]
            quest_line = quest_meta[Q_ID-1]
            if quest_line["id"] != Q_ID:
                print(f"Error at {Q_ID}")
                exit()
            file_name = quest_line['file_name']
            YesorNo = quest_line['YesorNo']
            QA_key = quest_line['QA_key']
            meta[file_name]["QA"][QA_key][YesorNo] = Q_line
            A_type = 'Ar' if YesorNo == 'Qr' else 'Aw'
            meta[file_name]["QA"][QA_key][A_type] = A_line
    return meta


def query(args):

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    print("tokenizer, model, image_processor are all prepared...")
    
    QA_meta_list = myprompt.load_meta()
    file_idx = 1
    for QA_path in QA_meta_list:
        print(f'No. {file_idx}: ' + QA_path)
        file_idx += 1
        Q_chart_path, A_chart_path, meta = build_question_image_json(QA_path)
        args.question_file = Q_chart_path
        args.answers_file = A_chart_path
        answer_path = QA_path.replace('QA', SAVE_ROOT)
        answer_path = answer_path.replace('meta.json', '')
        os.makedirs(answer_path, exist_ok=True)
        answer_path = os.path.join(answer_path, f'{MODEL_NAME}.json')
        # if os.path.exists(answer_path): continue
        
        query_once(args, tokenizer, image_processor, model, model_name)
        
        meta = prase_answer_json(Q_chart_path, A_chart_path, meta)
        with open(answer_path, 'w', encoding='utf-8') as fj:
            fj.write(json.dumps(meta, indent=4, ensure_ascii=False))
        # exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=CKPT_PATH)
    parser.add_argument("--model-base", type=str, default='/data/FinAi_Mapping_Knowledge/qiyiyan/models/llava-v1.5-13b')
    
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    query(args)
