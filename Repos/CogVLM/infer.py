import os
import sys, copy, argparse
sys.path.append('../')
import torch
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor
from utils.chat import chat
from sat.model.mixins import CachedAutoregressiveMixin
# import bitsandbytes
from PIL import Image

from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/cogvlm-chat'
TOKENIZER_PATH = '/path/to/models/vicuna-7b-v1.5'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/CogVLM.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        model, model_args = CogVLMModel.from_pretrained(
            CKPT_PATH,
            args=argparse.Namespace(
                deepspeed=None,
                local_rank=0,
                rank=0,
                world_size=1,
                model_parallel_size=1,
                mode='inference',
                skip_init=True,
                fp16=False,
                bf16=True,
                use_gpu_initialization=True,
                device='cuda',
            ))
        self.model = model.eval()
        self.tokenizer = llama2_tokenizer(TOKENIZER_PATH, signal_type="chat")
        self.image_processor = get_image_processor(model_args.eva_args["image_size"][0])
        self.model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
        self.text_processor_infer = llama2_text_processor_inference(self.tokenizer, None, model.image_length)
        
    def model_gen(self, question, im_path):
        with torch.no_grad():
            answer, _, _ = chat(
                im_path, 
                self.model,
                self.text_processor_infer,
                self.image_processor,
                question, 
                history=[],
                max_length=2048, 
                top_p=0.4, 
                temperature=0.8,
                top_k=1,
                invalid_slices=self.text_processor_infer.invalid_slices,
                no_prompt=False
                )
        return answer


if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
