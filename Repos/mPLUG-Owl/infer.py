import os
import sys, copy
sys.path.append('../')
import random, json, torch, time
from PIL import Image
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/mplug-owl-bloomz-7b-multilingual'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/mPLUG-Owl.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        model = MplugOwlForConditionalGeneration.from_pretrained(
            CKPT_PATH,
            torch_dtype=torch.bfloat16,
        ).to('cuda')
        image_processor = MplugOwlImageProcessor.from_pretrained(CKPT_PATH)
        tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH)
        processor = MplugOwlProcessor(image_processor, tokenizer)
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.processor = processor
        
    def model_gen(self, question, im_path):
        image_list = [im_path]
        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 512
        }
        images = [Image.open(_) for _ in image_list]
        inputs = self.processor(text=question, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        answer = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        return answer


if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
