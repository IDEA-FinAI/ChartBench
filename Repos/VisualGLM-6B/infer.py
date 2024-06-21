import os
import sys, copy
sys.path.append('../')
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/visualglm-6b'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/VisualGLM.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(CKPT_PATH, trust_remote_code=True).half().cuda()
        self.model = model.eval()
        self.tokenizer = tokenizer

    def model_gen(self, question, im_path):
        answer, _ = self.model.chat(self.tokenizer, im_path, question, history=[])
        return answer

if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
