import os
import sys, copy
sys.path.append('../')
import torch
from PIL import Image
import random, json, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/Qwen-VL-Chat'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/Qwen-VL-Chat.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(CKPT_PATH, device_map="cuda", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(CKPT_PATH, trust_remote_code=True)
        self.model = model
        self.tokenizer = tokenizer
        
    def model_gen(self, question, im_path):
        query = self.tokenizer.from_list_format([
            {'image': im_path},
            {'text': question},
        ])

        answer, _ = self.model.chat(self.tokenizer, query=query, history=None)
        return answer

if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
