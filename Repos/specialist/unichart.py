'''NOTE
conda activate xzz_2.0
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/Repos/specialist
CUDA_VISIBLE_DEVICES=7 python unichart.py
'''
import os
import sys, copy
sys.path.append('../')
import torch
from PIL import Image

from utils import ChartBenchTester, sys_prompt
from transformers import DonutProcessor, VisionEncoderDecoderModel

CKPT_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/qbw/cache/ckpt/unichart-chartqa-960'
SAVE_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/Result/raw/unichart-chartqa-960.json'
TEST_INDEX = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/test.jsonl'
IMG_ROOT = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/data'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionEncoderDecoderModel.from_pretrained(CKPT_PATH).to(self.device)
        self.processor = DonutProcessor.from_pretrained(CKPT_PATH)
        
    def model_gen(self, question, im_path):
        image = Image.open(im_path).convert('RGB')
        decoder_input_ids = self.processor.tokenizer(question, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generate_ids = self.model.generate(pixel_values.to(self.device),decoder_input_ids=decoder_input_ids.to(self.device), num_beams=4, max_new_tokens=512,pad_token_id=self.processor.tokenizer.pad_token_id,eos_token_id=self.processor.tokenizer.eos_token_id)
        answer = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        answer = answer.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        answer = answer.split("<s_answer>")[1].strip()
        # print(answer)
        return answer


if __name__ == '__main__':
    yn_prompt = '<chartqa> {} Please answer yes or no. <s_answer>'
    qa_prompt = '<chartqa> {} Answer the question using a single word or phrase. <s_answer>'
    tester = CustomChartBenchTester(
        TEST_INDEX,
        yn_prompt,
        qa_prompt
    )
    tester.reset_image_root(IMG_ROOT)
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
