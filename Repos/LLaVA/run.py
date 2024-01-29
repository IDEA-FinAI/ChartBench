import os

import sys, copy
sys.path.append('../')
import myprompt

import torch, json, time

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from tqdm import tqdm


MODEL_NAME = 'llava-v1.5-13b'
NOW_ROOT = myprompt.now_root
SAVE_ROOT = f'Eval/{myprompt.task_name}'
CKPT_PATH = f'/data/FinAi_Mapping_Knowledge/qiyiyan/models/{MODEL_NAME}'

def query_once(tokenizer, model, image_processor, image, question):
    
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, None)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    Q_base = copy.deepcopy(myprompt.prompt_yes_or_no)
    Q_base = Q_base.format(question)

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            Q_base = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + Q_base
        else:
            Q_base = DEFAULT_IMAGE_TOKEN + '\n' + Q_base
        conv.append_message(conv.roles[0], Q_base)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], Q_base)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=512,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    answer = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace('</s>', '')
    # conv.messages[-1][-1] = outputs
    return Q_base, answer


def query():
    load_8bit = True
    load_4bit = False
    device = 'cuda'
    disable_torch_init()
    model_name = get_model_name_from_path(CKPT_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(CKPT_PATH, None, model_name, load_8bit, load_4bit, device=device)
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
                    DIY_Qr, DIY_Ar = query_once(tokenizer, model, image_processor, raw_image, Qr)
                    DIY_Qw, DIY_Aw = query_once(tokenizer, model, image_processor, raw_image, Qw)
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