'''NOTE
conda activate intern_clean
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/Repos/Ours
CUDA_VISIBLE_DEVICES=4 python infer.py
'''
import os
import sys, copy
sys.path.append('../')
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision

from utils import ChartBenchTester, sys_prompt

CKPT_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/Repo/InternLM-XComposer/finetune/output/align_800k_vitproj_lora_sft_chartqa_chartbench_mmc_1epoch'
SAVE_PATH = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/Result/raw/Ours.jsonl'
TEST_INDEX = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/test.jsonl'
IMG_ROOT = '/data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartLLM/ChartBench/data'

def __padding__(image):
    width, height = image.size
    tar = max(width, height)
    top_padding = int((tar - height)/2)
    bottom_padding = tar - height - top_padding
    left_padding = int((tar - width)/2)
    right_padding = tar - width - left_padding
    image = torchvision.transforms.functional.pad(image, [left_padding, top_padding, right_padding, bottom_padding])
    return image

class CustomChartBenchTester(ChartBenchTester):

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(CKPT_PATH, device_map="cuda", trust_remote_code=True).eval().cuda().half()
        model.tokenizer = tokenizer
        self.model = model
        self.tokenizer = tokenizer

    def model_gen(self, question, im_path):
        padding = False
        need_bos = True
        pt1 = 0
        embeds = []
        im_mask = []
        images = [im_path]
        images_loc = [0]
        for i, pts in enumerate(images_loc + [len(question)]):
            subtext = question[pt1:pts]
            if need_bos or len(subtext) > 0:
                text_embeds = self.model.encode_text(subtext, add_special_tokens=need_bos)
                embeds.append(text_embeds)
                im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
                need_bos = False
            if i < len(images):
                try:
                    image = Image.open(images[i]).convert('RGB')
                except:
                    image = images[i].convert('RGB')
                if padding:
                    image = __padding__(image)
                image = self.model.vis_processor(image).unsqueeze(0).cuda()
                image_embeds = self.model.encode_img(image)
                embeds.append(image_embeds)
                im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
            pt1 = pts
        embeds = torch.cat(embeds, dim=1)
        im_mask = torch.cat(im_mask, dim=1)
        im_mask = im_mask.bool()

        outputs = self.model.generate(inputs_embeds=embeds, im_mask=im_mask,
                            temperature=1.0, max_new_tokens=500, num_beams=3,
                            do_sample=False, repetition_penalty=1.0)

        output_token = outputs[0]
        if output_token[0] == 0 or output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.model.tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('[UNUSED_TOKEN_145]')[0].strip()
        return output_text


if __name__ == '__main__':
    tester = CustomChartBenchTester(
        TEST_INDEX,
        sys_prompt["blip2 style"],
        sys_prompt["chartqa"]
    )
    tester.reset_image_root(IMG_ROOT)
    tester.load_model()
    tester.infer_all_answers(SAVE_PATH)
