'''NOTE
conda activate chartgemma
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/Repos/chartgemma
CUDA_VISIBLE_DEVICES=0 python infer.py 2>&1 | tee -a ./log/debug.log 
'''
import os, io
import sys, copy
sys.path.append('../')
import torch
from PIL import Image
import random, json, time
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/models/chartgemma'
IMG_ROOT = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/data'
TEST_INDEX = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/test.jsonl'
SAVE_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/Result/raw/chartgemma-pot.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(CKPT_PATH, torch_dtype=torch.float16).to(self.device)
        self.processor = AutoProcessor.from_pretrained(CKPT_PATH)
        
    def model_gen(self, question, im_path):
        image = Image.open(im_path).convert('RGB')
        inputs = self.processor(text=question, images=image, return_tensors="pt")
        prompt_length = inputs['input_ids'].shape[1]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generate_ids = self.model.generate(**inputs, num_beams=4, max_new_tokens=512)
        response = self.processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        def _execute_python_code(code):
            # 重定向 stdout 到一个 StringIO 对象
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout

            status = True
            try:
                # 执行传入的 Python 代码
                exec(code)
            except Exception as e:
                # 捕获执行中的异常并打印
                status = False
            finally:
                # 恢复 stdout
                sys.stdout = old_stdout

            # 获取打印输出的内容
            if status:
                output = new_stdout.getvalue()
            else:
                output = None
            return output, status
        
        if 'Program of Thought' in question:
            response, status = _execute_python_code(response)
            if status:
                answer = response
                # print(answer)
            else:
                answer = ""
                # print("error running...")
        else:
            answer = response
            answer = answer.replace("Supports","Yes").replace("Refutes","No")
            answer = answer.strip()
        return answer


if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['acc_pot'],
        sys_prompt_nqa=sys_prompt['nqa_pot']
    )
    tester.load_model()
    tester.reset_image_root(IMG_ROOT)
    tester.infer_all_answers(SAVE_PATH)
