import os
import sys, copy
sys.path.append('../')
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/blip2-flan-t5-xxl'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/BLIP2.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        processor = Blip2Processor.from_pretrained(CKPT_PATH)
        model = Blip2ForConditionalGeneration.from_pretrained(CKPT_PATH, torch_dtype=torch.float16, device_map='cuda')
        self.model = model.eval()
        self.processor = processor
        
    def model_gen(self, question, im_path):
        raw_image = Image.open(im_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to('cuda')
        out = self.model.generate(**inputs, max_new_tokens=4000)
        answer = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return answer

if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
