import os

import sys, copy
sys.path.append('../')
import myprompt

import random, json, torch, time
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

from tqdm import tqdm
from PIL import Image


MODEL_NAME = 'mplug-owl-bloomz-7b-multilingual'
NOW_ROOT = myprompt.now_root
SAVE_ROOT = f'Eval/{myprompt.task_name}'
CKPT_PATH = f'/data/FinAi_Mapping_Knowledge/qiyiyan/models/{MODEL_NAME}'

def query_once(tokenizer, processor, model, raw_image, question):
    
    # We use a human/AI template to organize the context as a multi-turn conversation.
    # <image> denotes an image placeholder.
    Q_base = copy.deepcopy(myprompt.prompt_yes_or_no)
    Q_base = [Q_base.format(question)]
    # The image paths should be placed in the image_list and kept in the same order as in the prompts.
    # We support urls, local file paths, and base64 string. You can customise the pre-processing of images by modifying the mplug_owl.modeling_mplug_owl.ImageProcessor
    image_list = [raw_image]

    # generate kwargs (the same in transformers) can be passed in the do_generate()
    generate_kwargs = {
        'do_sample': True,
        'top_k': 5,
        'max_length': 512
    }

    images = [Image.open(_) for _ in image_list]
    inputs = processor(text=Q_base, images=images, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    answer = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    return Q_base, answer


def query():
    model = MplugOwlForConditionalGeneration.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    image_processor = MplugOwlImageProcessor.from_pretrained(CKPT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH)
    processor = MplugOwlProcessor(image_processor, tokenizer)

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
                    DIY_Qr, DIY_Ar = query_once(tokenizer, processor, model, image_dir, Qr)
                    DIY_Qw, DIY_Aw = query_once(tokenizer, processor, model, image_dir, Qw)
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