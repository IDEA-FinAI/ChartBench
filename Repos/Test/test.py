import os
import sys, copy
sys.path.append('../')
from utils import ChartBenchTester

class CustomChartBenchTester(ChartBenchTester):
    def load_model(self):
        self.model = None
        self.processor = None
        
    def model_gen(self, question, im_path):
        return "Yes"

system_prompt_acc = 'user\nPlease determine whether the judgments on this chart are correct or not. Answer the question with yes or no.{}\nassistant\n'
system_prompt_nqa = 'user\nAnswer the question using a single word or phrase.{}\nassistant\n'
tester = CustomChartBenchTester(
    test_index= '/Users/sincerexu/Desktop/ChartBench/test.jsonl',
    sys_prompt_acc=system_prompt_acc,
    sys_prompt_nqa=system_prompt_nqa
)

save_path = '/Users/sincerexu/Desktop/ChartBench/Result/raw/BLIP2.jsonl'
tester.infer_all_answers(save_path)
