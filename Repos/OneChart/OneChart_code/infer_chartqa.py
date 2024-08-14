import os
import sys, copy

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

import sys
sys.path.append('/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartQA/Repos')
from utils import ChartQATester, evaluate_relaxed_accuracy
import warnings
warnings.filterwarnings("ignore")

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

CKPT_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/models/OneChart'
LLM_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/models/llava-v1.6-vicuna-13b'
SAVE_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartQA/Result/OneChart'

def list_json_value(json_dict):
    rst_str = []
    sort_flag = True
    try:
        for key, value in json_dict.items():
            if isinstance(value, dict):
                decimal_out = list_json_value(value)
                rst_str = rst_str + decimal_out
                sort_flag = False
            elif isinstance(value, list):
                return []
            else:
                if isinstance(value, float) or isinstance(value, int):
                    rst_str.append(value)
                else:
                    value = re.sub(r'\(\d+\)|\[\d+\]', '', value)
                    num_value = re.sub(r'[^\d.-]', '', str(value)) 
                    if num_value not in ["-", "*", "none", "None", ""]:
                        rst_str.append(float(num_value))
    except Exception as e:
        print(f"Error: {e}")
        # print(num_value)
        print(json_dict)
        return []
    # if len(rst_str) > 0:
    #     rst_str = rst_str + [float(-1)]
    return rst_str

def norm_(rst_list):
    if len(rst_list) < 2:
        return rst_list
    min_vals = min(rst_list)
    max_vals = max(rst_list)
    rst_list = np.array(rst_list)
    normalized_tensor = (rst_list - min_vals) / (max_vals - min_vals + 1e-9)
    return list(normalized_tensor)

class CustomChartQATester(ChartQATester):
    
    def load_model(self):
        disable_torch_init()
        model_name = os.path.expanduser(CKPT_PATH)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="right")
        model = varyOPTForCausalLM.from_pretrained(model_name)
        model.to(device='cuda',  dtype=torch.bfloat16)

        image_processor_high = test_transform
        use_im_start_end = True
        image_token_len = 256
        onechart_query = "Covert the key information of the chart to a python dict:"
        onechart_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + onechart_query + '\n'
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
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor_high = image_processor_high
        self.onechart_qs = onechart_qs
        self.model_name = model_name
        self.llava_conv_mode = llava_conv_mode
        self.llava_tokenizer = llava_tokenizer
        self.llava_model = llava_model
        self.llava_image_processor = llava_image_processor
        
    def get_pydict(self, im_path):
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], self.onechart_qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])
        image = image = Image.open(im_path).convert('RGB')
        image_1 = image.copy()
        image_tensor_1 = self.image_processor_high(image_1).to(torch.bfloat16)

        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stop_str = '</s>'
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): # bfloat16
            output_ids = self.model.generate(
                input_ids,
                images=[(image_tensor_1.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).cuda())],
                do_sample=False,
                num_beams = 1,
                # streamer=streamer,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria]
            )
        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        outputs = outputs.replace("<Number> ", "")
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        if outputs[-1] == '.':
            outputs = outputs[:-1]
        
        pred_nums = self.model.pred_locs
        ### judge reliable
        reliable_distence = 100
        is_reliable = False
        try:
            outputs_json = json.loads(outputs)
            list_v = list_json_value(outputs_json['values'])
            list_v = [round(x,4) for x in norm_(list_v)]
            gt_nums = torch.tensor(list_v).reshape(1,-1)
            # print("<Chart>: ", pred_nums[:len(list_v)])
            pred_nums_ = torch.tensor(pred_nums[:len(list_v)]).reshape(1,-1)
            reliable_distence = F.l1_loss(pred_nums_, gt_nums)
            # print("reliable_distence: ", reliable_distence)
            if reliable_distence < 0.1:
                is_reliable = True
                # print("After OneChart checking, this prediction is reliable.")
            else:
                is_reliable = False
                # print("This prediction may be has error! ")
        except Exception as e:
            is_reliable = False
            # print("This prediction may be has error! ")
        return outputs, reliable_distence, is_reliable

    def model_gen(self, question, im_path, py_dict):

        conv = conv_templates[self.llava_conv_mode].copy()
        if "mpt" in self.model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        image = image = Image.open(im_path).convert('RGB')
        
        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.llava_image_processor, None)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
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

        os.makedirs(output_path, exist_ok=True)
        print("Result will be saved at:")
        print(output_path)
        
        hint = 'The key information in the chart has been extracted as below:\n{}\n'
        
        part_acc = []
        for part_name in ['human', 'augmented']:
            part_json = os.path.join(output_path, f"{part_name}.json")
            if os.path.exists(part_json):
                print(f"Load result from: {part_json}")
                part = json.load(open(part_json, 'r'))
            else:
                part = []
                samples = json.load(open(self.root+f'test/test_{part_name}.json')) 
                for q in tqdm(samples):
                    im_path = self.root + 'test/png/'+q['imgname']
                    py_dict, _, _ = self.get_pydict(im_path)
                    text = hint.format(py_dict) + self.system_prompt.format(q['query'])
                    with torch.cuda.amp.autocast():
                        response = self.model_gen(text, im_path, py_dict)
                        # response = self.model_gen(text, im_path)  
                    part.append({
                        'image': im_path,
                        'query': text,
                        'answer': response,
                        'annotation': q['label'] 
                    }) 
                with open(part_json, 'w') as f:
                    json.dump(part, f, indent=4)
            part_acc.append(part)
        
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["@AP", "0.05", "0.1", "0.2"]  
        human_row = ["Human"]
        augmented_row = ["Augmented"]
        averaged_row = ["Averaged"]
        for ap in [0.05, 0.1, 0.2]:
            part_acc_ap = [evaluate_relaxed_accuracy(p, self.metric, ap) for p in part_acc]
            human_acc = part_acc_ap[0]
            augmented_acc = part_acc_ap[1]
            averaged_acc = (human_acc + augmented_acc) / 2
            human_row.append(human_acc)
            augmented_row.append(augmented_acc)
            averaged_row.append(averaged_acc)
     
        table.add_row(human_row)
        table.add_row(augmented_row)
        table.add_row(averaged_row)

        table_path = os.path.join(output_path, 'table.txt')
        with open(table_path, 'w') as f:
            f.write(str(table))

        print(table)

  
if __name__ == '__main__':
    tester = CustomChartQATester()
    tester.load_model()
    tester.reset_prompt("Answer the question using a single word or phrase. {}")
    # tester.reset_metric("relaxed_acc_xzz")
    tester.infer_all_answers(SAVE_PATH)
