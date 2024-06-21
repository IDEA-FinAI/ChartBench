import os
import sys, copy
sys.path.append('../')
import random, json, time
import torch
import requests
from PIL import Image
from io import BytesIO
from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mgm.conversation import conv_templates, SeparatorStyle
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from utils import sys_prompt, ChartBenchTester

CKPT_PATH = '/path/to/models/MGM/MGM-13B-HD'
TEST_INDEX = '/path/to/ChartBench/test.jsonl'
SAVE_PATH = '/path/to/ChartBench/Result/raw/Mini-Gemini.jsonl'

class CustomChartBenchTester(ChartBenchTester):
    
    def load_model(self):
        model_name = get_model_name_from_path(CKPT_PATH)
        tokenizer, model, image_processor, context_len = load_pretrained_model(CKPT_PATH, None, model_name, False, False, device='cuda')
        if '8x7b' in model_name.lower():
            conv_mode = "mistral_instruct"
        elif '34b' in model_name.lower():
            conv_mode = "chatml_direct"
        elif '2b' in model_name.lower():
            conv_mode = "gemma"
        else:
            conv_mode = "vicuna_v1"
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.image_processor = image_processor
        self.conv_mode = conv_mode
        
    def model_gen(self, question, im_path):

        conv = conv_templates[self.conv_mode].copy()
        # if "mpt" in self.model_name.lower():
        #     roles = ('user', 'assistant')
        # else:
        #     roles = conv.roles
        images = [im_path]
        image_convert = []
        for _image in images:
            image_convert.append(Image.open(_image).convert('RGB'))
        if hasattr(self.model.config, 'image_size_aux'):
            if not hasattr(self.image_processor, 'image_size_raw'):
                self.image_processor.image_size_raw = self.image_processor.crop_size.copy()
            self.image_processor.crop_size['height'] = self.model.config.image_size_aux
            self.image_processor.crop_size['width'] = self.model.config.image_size_aux
            self.image_processor.size['shortest_edge'] = self.model.config.image_size_aux
        # Similar operation in model_worker.py
        image_tensor = process_images(image_convert, self.image_processor, self.model.config)
        image_grid = getattr(self.model.config, 'image_grid', 1)
        if hasattr(self.model.config, 'image_size_aux'):
            raw_shape = [self.image_processor.image_size_raw['height'] * image_grid,
                        self.image_processor.image_size_raw['width'] * image_grid]
            image_tensor_aux = image_tensor 
            image_tensor = torch.nn.functional.interpolate(image_tensor,
                                                        size=raw_shape,
                                                        mode='bilinear',
                                                        align_corners=False)
        else:
            image_tensor_aux = []
        if image_grid >= 2:            
            raw_image = image_tensor.reshape(3, 
                                            image_grid,
                                            self.image_processor.image_size_raw['height'],
                                            image_grid,
                                            self.image_processor.image_size_raw['width'])
            raw_image = raw_image.permute(1, 3, 0, 2, 4)
            raw_image = raw_image.reshape(-1, 3,
                                        self.image_processor.image_size_raw['height'],
                                        self.image_processor.image_size_raw['width'])
                    
            if getattr(self.model.config, 'image_global', False):
                global_image = image_tensor
                if len(global_image.shape) == 3:
                    global_image = global_image[None]
                global_image = torch.nn.functional.interpolate(global_image, 
                                                            size=[self.image_processor.image_size_raw['height'],
                                                                    self.image_processor.image_size_raw['width']], 
                                                            mode='bilinear', 
                                                            align_corners=False)
                # [image_crops, image_global]
                raw_image = torch.cat([raw_image, global_image], dim=0)
            image_tensor = raw_image.contiguous()
            image_tensor = image_tensor.unsqueeze(0)
        
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
            image_tensor_aux = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor_aux]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            image_tensor_aux = image_tensor_aux.to(self.model.device, dtype=torch.float16)

        try:
            inp = question
        except EOFError:
            inp = ""
        if images is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = (DEFAULT_IMAGE_TOKEN + '\n')*len(images) + inp
            conv.append_message(conv.roles[0], inp)
            images = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # add image split string
        if prompt.count(DEFAULT_IMAGE_TOKEN) >= 2:
            final_str = ''
            sent_split = prompt.split(DEFAULT_IMAGE_TOKEN)
            for _idx, _sub_sent in enumerate(sent_split):
                if _idx == len(sent_split) - 1:
                    final_str = final_str + _sub_sent
                else:
                    final_str = final_str + _sub_sent + f'Image {_idx+1}:' + DEFAULT_IMAGE_TOKEN
            prompt = final_str
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                images_aux=image_tensor_aux if len(image_tensor_aux)>0 else None,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512,
                bos_token_id=self.tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=self.tokenizer.pad_token_id,  # Pad token
                # streamer=streamer,
                use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # conv.messages[-1][-1] = outputs

        return outputs

if __name__ == '__main__':
    tester = CustomChartBenchTester(
        test_index=TEST_INDEX,
        sys_prompt_acc=sys_prompt['blip2 style'],
        sys_prompt_nqa=sys_prompt['chartqa']
    )
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
