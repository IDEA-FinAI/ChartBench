import os
import sys, copy, random, argparse
sys.path.append('../')

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/MiniGPT-4/ckpts/minigptv2_checkpoint.pth'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/MiniGPT-v2.jsonl'

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

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        cudnn.benchmark = False
        cudnn.deterministic = True

        args = parse_args()
        cfg = Config(args)
        device = 'cuda:{}'.format(args.gpu_id)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        self.model = model.eval()
        self.chat = Chat(model, vis_processor, device=device)
        self.vis_processor = vis_processor
        
    def model_gen(self, question, im_path):
        chat_state = CONV_VISION.copy()
        img_list = []
        raw_image = Image.open(im_path).convert('RGB')
        _ = self.chat.upload_img(raw_image, chat_state, img_list)
        
        self.chat.ask(question, chat_state)
        if len(img_list) > 0:
            if not isinstance(img_list[0], torch.Tensor):
                self.chat.encode_img(img_list)
        streamer = self.chat.stream_answer(
            conv=chat_state,
            img_list=img_list,
            temperature=0.6,
            max_new_tokens=500,
            max_length=2000
        )
        output = ''
        for new_output in streamer:
            escapped = escape_markdown(new_output)
            output += escapped
        chat_state.messages[-1][1] = '</s>'
        answer = output
        return answer

if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
