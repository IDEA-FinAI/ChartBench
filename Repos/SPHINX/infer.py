import os
import sys, copy
sys.path.append('../')

import torch
import torch.distributed as dist
import multiprocessing as mp
from PIL import Image
from SPHINX.sphinx import SPHINXModel
from utils import sys_prompt, ChartBenchTester

import warnings
warnings.filterwarnings("ignore")

CKPT_PATH = '/path/to/models/sphinx/finetune/mm/SPHINX/SPHINX'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/SPHINX.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self, rank, world_size):
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size,
            init_method=f"tcp://127.0.0.1:23560",
        )
        torch.cuda.set_device(rank)
        model = SPHINXModel.from_pretrained(
            pretrined_path=CKPT_PATH, 
            with_visual=True,
            mp_group=dist.new_group(ranks=list(range(world_size)))
        )
        self.model = model.eval()

    def model_gen(self, question, im_path):
        image = Image.open(im_path)
        qas = [[question, None]]
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                answer = self.model.generate_reponse(
                    qas, image, max_gen_len=1024, 
                    temperature=0.9, top_p=0.5, seed=0)
        answer = answer.strip('\n')
        return answer
    
def main_worker(world_size, rank):
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model(rank, world_size)
    tester.infer_all_answers(SAVE_PATH)

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    N_GPU = 2
    for rank in range(N_GPU):
        process = mp.Process(target=main_worker, args=(N_GPU, rank))
        process.start()
