'''NOTE
conda activate xzz_2.0
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/Repos/specialist
CUDA_VISIBLE_DEVICES=6 python matcha.py
'''
import os
import sys, copy
sys.path.append('../')
import torch
from PIL import Image

from utils import ChartBenchTester, sys_prompt
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

CKPT_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/qbw/cache/ckpt/matcha-chartqa'
SAVE_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/Result/raw/matcha-chartqa.json'
TEST_INDEX = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/test.jsonl'
IMG_ROOT = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/data'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Pix2StructForConditionalGeneration.from_pretrained(CKPT_PATH).to(self.device)
        self.processor = Pix2StructProcessor.from_pretrained(CKPT_PATH)
        self.processor.image_processor.is_vqa = True
        
    def model_gen(self, question, im_path):
        image = Image.open(im_path).convert('RGB')
        inputs = self.processor(text=question, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generate_ids = self.model.generate(**inputs, num_beams=4, max_new_tokens=512)
        answer = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(answer)
        return answer


if __name__ == '__main__':
    tester = CustomChartBenchTester(
        TEST_INDEX,
        sys_prompt["blip2 style"],
        sys_prompt["chartqa"]
    )
    tester.reset_image_root(IMG_ROOT)
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
