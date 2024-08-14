import os
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
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
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

CKPT_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/models/deplot'
LLM_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/models/llava-v1.6-vicuna-13b'
SAVE_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartQA/Result/Deplot'

class CustomChartQATester(ChartQATester):
    
    def load_model(self):
        disable_torch_init()

        self.model = Pix2StructForConditionalGeneration.from_pretrained(CKPT_PATH)
        self.processor = AutoProcessor.from_pretrained(CKPT_PATH)

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
        self.llava_conv_mode = llava_conv_mode
        self.llava_tokenizer = llava_tokenizer
        self.llava_model = llava_model
        self.llava_image_processor = llava_image_processor
        
    def get_pydict(self, im_path):
        img = Image.open(im_path)
        inputs = self.processor(images=img, text="Generate underlying data table of the figure below:", return_tensors="pt")
        predictions = self.model.generate(**inputs, max_new_tokens=512)
        answer = self.processor.decode(predictions[0], skip_special_tokens=True)
        return answer

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
                    py_dict = self.get_pydict(im_path)
                    text = hint.format(py_dict) + self.system_prompt.format(q['query'])
                    with torch.cuda.amp.autocast():
                        response = self.model_gen(text, im_path)
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