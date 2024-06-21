import os
import sys, copy
sys.path.append('../')
import torch, json, time
from transformers import AutoModel, AutoTokenizer
from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/internlm-xcomposer-7b'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/InternLM-XComposer.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        torch.set_grad_enabled(False)
        self.model = AutoModel.from_pretrained(CKPT_PATH, trust_remote_code=True).cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH, trust_remote_code=True)
        self.model.tokenizer = self.tokenizer
        
    def model_gen(self, question, im_path):
        answer = self.model.generate(question, im_path).strip('\n')
        return answer


if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
