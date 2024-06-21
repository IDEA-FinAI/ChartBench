import os
import sys, copy
sys.path.append('../')

import torch, json, time

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/llava-v1.5-13b'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/LLaVA.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        load_8bit = True
        load_4bit = False
        device = 'cuda'
        disable_torch_init()
        model_name = get_model_name_from_path(CKPT_PATH)
        tokenizer, model, image_processor, _ = load_pretrained_model(CKPT_PATH, None, model_name, load_8bit, load_4bit, device=device)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        
    def model_gen(self, question, im_path):
        conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        image = Image.open(im_path).convert('RGB')
        
        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.image_processor, None)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                question = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv.append_message(conv.roles[0], question)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], question)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        answer = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace('</s>', '')
        return answer


if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
