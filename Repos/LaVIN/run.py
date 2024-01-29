import os

import sys, copy
sys.path.append('../')
import myprompt

from PIL import Image
import random, json, time
from tqdm import tqdm
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from util.misc import get_rank
# from minigpt4.common.registry import registry
from conversation.conversation import Chat, CONV_VISION
from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from eval import load
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from typing import Tuple


MODEL_NAME = 'LaVIN'
NOW_ROOT = myprompt.now_root
SAVE_ROOT = f'Eval/{myprompt.task_name}'
LLAMA_PATH = "/data/FinAi_Mapping_Knowledge/qiyiyan/models/llama-13b"


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def setup_seeds(seed):
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def query_once(model, raw_image, question):
    Q_base = copy.deepcopy(myprompt.prompt_yes_or_no)
    Q_base = Q_base.format(question)
    chat_state = CONV_VISION.copy()
    img_list = []
    answer = model.upload_img(raw_image, chat_state, img_list)
    model.ask(Q_base, chat_state)
    answer = model.answer(conv=chat_state,
                            img_list=img_list,
                            num_beams=1,
                            temperature=1,
                            max_new_tokens=300,
                            max_length=2000)
    return Q_base, answer


def query():
    local_rank, world_size = setup_model_parallel()
    lavin = load(
        ckpt_dir=LLAMA_PATH,
        llm_model="13B",
        adapter_path='./weight/llama13B-15-eph-conv.pth',
        max_seq_len=512,
        max_batch_size=4,
        adapter_type='attn',
        adapter_dim=8,
        adapter_scale=1,
        hidden_proj=128,
        visual_adapter_type='router',
        temperature=5.,
        tokenizer_path='',
        local_rank=local_rank,
        world_size=world_size,
        use_vicuna=False
    )

    vis_processor = transforms.Compose([
        transforms.Resize((224, 224), 
        interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    chat = Chat(lavin, vis_processor, device=torch.device('cuda'))
    
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
            file_list = list(meta.keys())[:5] # slow
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
                    DIY_Qr, DIY_Ar = query_once(chat, raw_image, Qr)
                    DIY_Qw, DIY_Aw = query_once(chat, raw_image, Qw)
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