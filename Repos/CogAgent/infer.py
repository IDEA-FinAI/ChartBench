import os
import sys, copy
sys.path.append('../')
import torch
import requests
import time
import json
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/CogAgent/CogAgent-vqa'
TOKENIZER_PATH = '/path/to/models/vicuna-7b-v1.5'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/CogAgent.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            CKPT_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            trust_remote_code=True
        ).eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def model_gen(self, question, im_path):
        raw_image = Image.open(im_path).convert('RGB')
        history = []
        input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=question, history=history, images=[raw_image])
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[input_by_model['images'][0].to(self.device).to(torch.bfloat16)]] if raw_image is not None else None,
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.device).to(torch.bfloat16)]]
        # add any transformers params here.
        gen_kwargs = {
            "max_length": 2048,
            "do_sample": False
        } # "temperature": 0.9
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]

        return response


if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
