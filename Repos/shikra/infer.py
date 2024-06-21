import os
import sys, copy
sys.path.append('../')

import logging
import argparse
import torch
from PIL import Image
from mmengine import Config
import transformers
from transformers import BitsAndBytesConfig
from mllm.dataset.process_function import PlainBoxFormatter
from mllm.dataset.builder import prepare_interactive
from mllm.models.builder.build_shikra import load_pretrained_shikra

from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/shikra-7b'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/Shikra.jsonl'

def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
class CustomChartBenchTester(ChartBenchTester):
    
    def load_args(self):
        log_level = logging.WARNING
        transformers.logging.set_verbosity(log_level)
        transformers.logging.enable_default_handler()
        transformers.logging.enable_explicit_format()

        parser = argparse.ArgumentParser("Shikra Web Demo")
        parser.add_argument('--load_in_8bit', action='store_true')
        parser.add_argument('--server_name', default=None)
        parser.add_argument('--server_port', type=int, default=None)
        args = parser.parse_args()

        self.model_args = Config(dict(
            type='shikra',
            version='v1',

            # checkpoint config
            cache_dir=None,
            model_name_or_path=CKPT_PATH,
            vision_tower=r'openai/clip-vit-large-patch14',
            pretrain_mm_mlp_adapter=None,

            # model config
            mm_vision_select_layer=-2,
            model_max_length=2048,

            # finetune config
            freeze_backbone=False,
            tune_mm_mlp_adapter=False,
            freeze_mm_mlp_adapter=False,

            # data process config
            is_multimodal=True,
            sep_image_conv_front=False,
            image_token_len=256,
            mm_use_im_start_end=True,

            target_processor=dict(
                boxes=dict(type='PlainBoxFormatter'),
            ),

            process_func_args=dict(
                conv=dict(type='ShikraConvProcess'),
                target=dict(type='BoxFormatProcess'),
                text=dict(type='ShikraTextProcess'),
                image=dict(type='ShikraImageProcessor'),
            ),

            conv_args=dict(
                conv_template='vicuna_v1.1',
                transforms=dict(type='Expand2square'),
                tokenize_kwargs=dict(truncation_size=None),
            ),

            gen_kwargs_set_pad_token_id=True,
            gen_kwargs_set_bos_token_id=True,
            gen_kwargs_set_eos_token_id=True,
        ))

        self.training_args = Config(dict(
            bf16=False,
            fp16=True,
            device='cuda',
            fsdp=None,
        ))

        self.quantization_kwargs = dict(
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
            )
        )


    def load_model(self):
        model, preprocessor = load_pretrained_shikra(self.model_args, self.training_args, self.quantization_kwargs)
        if not getattr(model, 'is_quantized', False):
            model.to(dtype=torch.float16, device=torch.device('cuda'))
        if not getattr(model.model.vision_tower[0], 'is_quantized', False):
            model.model.vision_tower[0].to(dtype=torch.float16, device=torch.device('cuda'))
        preprocessor['target'] = {'boxes': PlainBoxFormatter()}
        tokenizer = preprocessor['text']
        self.model = model
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        
    def model_gen(self, question, im_path):
        do_sample = False
        max_length = 2048
        ds = prepare_interactive(self.model_args, self.preprocessor)

        image = Image.open(im_path).convert("RGB")
        image = expand2square(image)
        ds.set_image(image)
        ds.append_message(role=ds.roles[0], message=question, boxes=[], boxes_seq=[])
        model_inputs = ds.to_model_input()
        model_inputs['images'] = model_inputs['images'].to(torch.float16)

        gen_kwargs = dict(
            use_cache=True,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_length,
        )
            
        input_ids = model_inputs['input_ids']
        with torch.inference_mode():
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                output_ids = self.model.generate(**model_inputs, **gen_kwargs)
        input_token_len = input_ids.shape[-1]
        answer = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        return answer


if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_args()
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
