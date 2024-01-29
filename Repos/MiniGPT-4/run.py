import os

import sys, copy
sys.path.append('../')
import myprompt


from PIL import Image
import random, json, time
from tqdm import tqdm
import argparse

import numpy as np
import torch

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


MODEL_NAME = 'minigpt_v2'
NOW_ROOT = myprompt.now_root
SAVE_ROOT = f'Eval/{myprompt.task_name}'
CKPT_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/models/MiniGPT-4/ckpts/minigptv2_checkpoint.pth'

def escape_markdown(text):
    # List of Markdown special characters that need to be escaped
    md_chars = ['<', '>']

    # Escape each special character
    for char in md_chars:
        text = text.replace(char, '\\' + char)

    return text

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cudnn.benchmark = False
cudnn.deterministic = True


CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)


def query_once(chat, raw_image, question):
    Q_base = copy.deepcopy(myprompt.prompt_yes_or_no)
    Q_base = Q_base.format(question)
    chat_state = CONV_VISION.copy()
    img_list = []
    _ = chat.upload_img(raw_image, chat_state, img_list)
    
    chat.ask(Q_base, chat_state)
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)
    streamer = chat.stream_answer(conv=chat_state,
                                img_list=img_list,
                                temperature=0.6,
                                max_new_tokens=500,
                                max_length=2000)
    output = ''
    for new_output in streamer:
        escapped = escape_markdown(new_output)
        output += escapped
    chat_state.messages[-1][1] = '</s>'
    answer = output
    # print(f"question: {Q_base}\n answer: {output}")
    # exit()
    return Q_base, answer


def query():
    print('Initializing Chat')
    
    args = parse_args()
    cfg = Config(args)
    device = 'cuda:{}'.format(args.gpu_id)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    model = model.eval()
    chat = Chat(model, vis_processor, device=device)

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
                    DIY_Qr, DIY_Ar = query_once(chat, image_dir, Qr)
                    DIY_Qw, DIY_Aw = query_once(chat, image_dir, Qw)
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