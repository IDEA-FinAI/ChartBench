import os
import sys, copy
sys.path.append('../')

import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/instructblip-vicuna-7b'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/InstructBLIP.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        self.model = InstructBlipForConditionalGeneration.from_pretrained(CKPT_PATH).to('cuda')
        self.processor = InstructBlipProcessor.from_pretrained(CKPT_PATH)
        
    def model_gen(self, question, im_path):
        raw_image = Image.open(im_path).convert('RGB')
        inputs = self.processor(images=raw_image, text=question, return_tensors="pt").to('cuda')
        outputs = self.model.generate(
                **inputs,
                do_sample=True,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        answer = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return answer


if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
