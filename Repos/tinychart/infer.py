import os
import sys, copy
sys.path.append('../')
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision

from utils import ChartBenchTester, sys_prompt

from prettytable import PrettyTable
import json
from tqdm import tqdm

from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model
from tinychart.eval.eval_metric import parse_model_output, evaluate_cmds

CKPT_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/qbw/cache/ckpt/mPLUG/TinyChart-3B-768'
SAVE_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/Result/raw/tinychart_new.jsonl'
TEST_INDEX = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/test.jsonl'
IMG_ROOT = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/data'

class CustomChartBenchTester(ChartBenchTester):

    def load_model(self):
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            CKPT_PATH, 
            model_base=None,
            model_name=get_model_name_from_path(CKPT_PATH),
            device="cuda"
        )
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.context_len = context_len
    
    def model_gen(self, question, im_path):
        answer = inference_model([im_path], question, self.model, self.tokenizer, self.image_processor, self.context_len, conv_mode="phi", max_new_tokens=500)
        return answer
    
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
                    
                Ar = self.model_gen(Qr, im_path)
                Aw = self.model_gen(Qw, im_path)
                
                samples[i]["conversation"][0]["query"] = Qr
                samples[i]["conversation"][1]["query"] = Qw
                samples[i]["conversation"][0]["answer"] = Ar
                samples[i]["conversation"][1]["answer"] = Aw

            if samples[i]["type"]["QA"] == "GPT-acc":
                Qr = self.system_prompt_nqa.format(samples[i]["conversation"][0]["query"])
                Ar = self.model_gen(Qr, im_path)
                samples[i]["conversation"][0]["query"] = Qr
                samples[i]["conversation"][0]["answer"] = Ar

            self.save_jsonl(output_path, [samples[i]], mode='a+')

if __name__ == '__main__':
    tester = CustomChartBenchTester(
        TEST_INDEX,
        "{} Answer Yes or No.",
        "{} Answer the question using a single word or phrase."
    )
    tester.reset_image_root(IMG_ROOT)
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)