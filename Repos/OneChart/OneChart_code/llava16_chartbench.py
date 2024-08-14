import os
import sys, copy
sys.path.append('../../')
import json
import numpy as np
import re
from tqdm import tqdm

from io import BytesIO
from transformers import TextStreamer
from vary.model.plug.transforms import test_transform
from vary.utils.conversation import conv_templates, SeparatorStyle
from vary.utils.utils import disable_torch_init
from transformers import StoppingCriteria
from vary.model import *
from vary.utils.utils import KeywordsStoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from utils import sys_prompt, ChartBenchTester
import warnings
warnings.filterwarnings("ignore")

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


LLM_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/models/llava-v1.6-vicuna-13b'
TEST_INDEX = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/test_toy.jsonl'
SAVE_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/Result/raw/llava16.jsonl'
IMG_ROOT = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/data'


class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        disable_torch_init()

        load_8bit = False
        load_4bit = True
        model_name = get_model_name_from_path(LLM_PATH)
        llava_tokenizer, llava_model, llava_image_processor, llava_context_len = load_pretrained_model(LLM_PATH, None, model_name, load_8bit, load_4bit, device='cuda')

        if 'llama-2' in model_name.lower():
            llava_conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            llava_conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            llava_conv_mode = "mpt"
        else:
            llava_conv_mode = "llava_v0"
            
        self.conv_mode = 'v1'
        self.model_name = model_name
        self.device = 'cuda'
        self.llava_conv_mode = llava_conv_mode
        self.llava_tokenizer = llava_tokenizer
        self.llava_model = llava_model
        self.llava_image_processor = llava_image_processor


    def model_gen(self, question, im_path):

        conv = conv_templates[self.llava_conv_mode].copy()
        if "mpt" in self.model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        image = image = Image.open(im_path).convert('RGB')
        
        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.llava_image_processor, None)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        inp = question
        if self.llava_model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.llava_tokenizer, input_ids)
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            output_ids = self.llava_model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512,
                # streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = self.llava_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        # conv.messages[-1][-1] = outputs
        return outputs.split("</s>")[0]

    def infer_all_answers(self, output_path):

        directory = os.path.dirname(output_path)
        os.makedirs(directory, exist_ok=True)
        samples = self.load_jsonl(self.test_index, mode='r')
        
        if os.path.exists(output_path): # ckpt
            ckpt_index = len(self.load_jsonl(output_path, mode='r'))
            print(f'Start from sample {ckpt_index} ...')
        else:
            ckpt_index = -1
        
        for i in tqdm(range(len(samples))):
            
            if samples[i]['id'] < ckpt_index: continue
            
            im_path = samples[i]["image"].replace('./data', self.image_root)
            if samples[i]["type"]["QA"] == "Acc+":
                Qr = self.system_prompt_acc.format(samples[i]["conversation"][0]["query"])
                Qw = self.system_prompt_acc.format(samples[i]["conversation"][1]["query"])
                with torch.cuda.amp.autocast():
                    Ar = self.model_gen(Qr, im_path)
                    Aw = self.model_gen(Qw, im_path)
                samples[i]["conversation"][0]["query"] = Qr
                samples[i]["conversation"][1]["query"] = Qw
                samples[i]["conversation"][0]["answer"] = Ar
                samples[i]["conversation"][1]["answer"] = Aw

            if samples[i]["type"]["QA"] == "GPT-acc":
                Qr = self.system_prompt_nqa.format(samples[i]["conversation"][0]["query"])
                with torch.cuda.amp.autocast():
                    Ar = self.model_gen(Qr, im_path)
                samples[i]["conversation"][0]["query"] = Qr
                samples[i]["conversation"][0]["answer"] = Ar

            self.save_jsonl(output_path, [samples[i]], mode='a+')
            
if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.reset_image_root(IMG_ROOT)
    tester.infer_all_answers(SAVE_PATH)
