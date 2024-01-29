import os

import sys, copy
sys.path.append('../')
import myprompt

from PIL import Image
import random, json, time
from tqdm import tqdm

from SPHINX.sphinx import SPHINXModel
from PIL import Image
import torch
import torch.distributed as dist
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = 'SPHINX'
NOW_ROOT = myprompt.now_root
SAVE_ROOT = f'Eval/{myprompt.task_name}'
CKPT_PATH = f'/data/FinAi_Mapping_Knowledge/qiyiyan/models/sphinx/finetune/mm/SPHINX/SPHINX'


def query_once(model, image_path, question):
    
    Q_base = copy.deepcopy(myprompt.prompt_yes_or_no)
    Q_base = Q_base.format(question)
    
    image = Image.open(image_path)
    qas = [[Q_base, None]]
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            answer = model.generate_reponse(qas, image, 
                                            max_gen_len=1024, 
                                            temperature=0.9, 
                                            top_p=0.5, 
                                            seed=0)
    answer = answer.strip('\n')
    return Q_base, answer


def main(world_size, rank):
    
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method=f"tcp://127.0.0.1:23560",
    )
    torch.cuda.set_device(rank)
    model = SPHINXModel.from_pretrained(pretrined_path=CKPT_PATH, 
                                        with_visual=True,
                                        mp_group=dist.new_group(ranks=list(range(world_size)))
                                        )
    
    QA_meta_list = myprompt.load_meta()
    logger = open('./log.txt', 'w')
    file_idx = 1
    for QA_path in QA_meta_list:
        if rank == 0: print(f'No. {file_idx}: ' + QA_path)
        file_idx += 1
        answer_path = QA_path.replace('QA', SAVE_ROOT)
        answer_path = answer_path.replace('meta.json', '')
        os.makedirs(answer_path, exist_ok=True)
        answer_path = os.path.join(answer_path, f'{MODEL_NAME}.json')
        if os.path.exists(answer_path): continue
        
        with open(QA_path, 'r') as fmeta:
            meta = json.load(fmeta)
            file_list = list(meta.keys())
            for file in tqdm(file_list):
                # if file == '21.txt': continue
                start_time = time.time()
                QAs = meta[file]["QA"]
                image_dir = meta[file]['image_path']
                image_dir = os.path.join(NOW_ROOT, image_dir)
                # raw_image = Image.open(image_dir).convert('RGB')
                for key in QAs.keys():
                    logger.write(image_dir + '\t' + key + '\n')
                    Qr = meta[file]["QA"][key]['Qr']
                    Qw = meta[file]["QA"][key]['Qw']
                    DIY_Qr, DIY_Ar = query_once(model, image_dir, Qr)
                    DIY_Qw, DIY_Aw = query_once(model, image_dir, Qw)
                    meta[file]["QA"][key]['Qr'] = DIY_Qr
                    meta[file]["QA"][key]['Ar'] = DIY_Ar
                    meta[file]["QA"][key]['Qw'] = DIY_Qw
                    meta[file]["QA"][key]['Aw'] = DIY_Aw
                end_time = time.time()
                run_time = end_time - start_time
                meta[file]["InfTime"] = str(run_time)

        if world_size > 1: torch.cuda.synchronize()
        with open(answer_path, 'w', encoding='utf-8') as fj:
            fj.write(json.dumps(meta, indent=4, ensure_ascii=False))
        # exit()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    N_GPU = 2
    for rank in range(N_GPU):
        process = mp.Process(target=main, args=(N_GPU, rank))
        process.start()
