import os
import sys, copy
sys.path.append('../../')
from docowl_infer import DocOwlInfer
from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/mPLUG/DocOwl1.5-Omni'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/DocOwl-v1.5.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        self.model = DocOwlInfer(ckpt_path=CKPT_PATH, anchors='grid_9', add_global_img=False)
        
    def model_gen(self, question, im_path):
        answer = self.model.inference(im_path, question)
        return answer

if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
