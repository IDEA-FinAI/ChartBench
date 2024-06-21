# ChartBench: A Benchmark for Complex Visual Reasoning in Charts

<a href='https://arxiv.org/abs/2312.15915'><img src='https://img.shields.io/badge/arXiv-2312.15915-b31b1b.svg'></a> <a href='https://github.com/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://github.com/buaacyw/GaussianEditor/blob/master/LICENSE.txt'><img src='https://img.shields.io/badge/License-MIT-blue'></a> [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/SincereX/ChartBench)


## Introduction

We propose the challenging ChartBench to evaluate the chart recognition of MLLMs.


![ChartBench Pipeline.](./asset/pipeline.png)

We improve the *Acc+* metric to avoid the randomly guessing situations.

![improved Acc+ metric.](./asset/QA.png)

We collect a larger set of unlabeled charts to emphasize the MLLM's ability to interpret visual information without the aid of annotated data points.

![Chart distributions and ChartCoT.](./asset/chart_distribution.png)


## Todo
- [ ] Open source: all evaluation results.
- [x] Open source: all data of ChartBench.
- [x] Open source: the evaluate scripts.
- [x] Open source: the inference scripts.
- [x] Open source: the demo data (10%).

## Setup
Please follow the official repository instructions below to set up the local environment.

<a href='https://huggingface.co/spaces/Salesforce/BLIP2'><img src='https://img.shields.io/badge/BLIP2-https://huggingface.co/spaces/Salesforce/BLIP2-blue'></a>
<a href='https://huggingface.co/docs/transformers/model_doc/instructblip'><img src='https://img.shields.io/badge/InstructBLIP-https://huggingface.co/docs/transformers/model_doc/instructblip-blue'></a>
<a href='https://github.com/THUDM/CogVLM'><img src='https://img.shields.io/badge/CogVLM-https://github.com/THUDM/CogVLM-blue'></a>
<a href='https://github.com/QwenLM/Qwen-VL'><img src='https://img.shields.io/badge/Qwen_VL_Chat-https://github.com/QwenLM/QwenVL-blue'></a>
<a href='https://llava-vl.github.io/'><img src='https://img.shields.io/badge/LLaVA_v1.5-https://llava_vl.github.io/-blue'></a>
<a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/MiniGPT_v2-https://github.com/VisionCAIR/MiniGPT4-blue'></a>
<a href='https://github.com/THUDM/VisualGLM-6B'><img src='https://img.shields.io/badge/VisualGLM-https://github.com/THUDM/VisualGLM6B-blue'></a>
<a href='https://github.com/X-PLUG/mPLUG-Owl'><img src='https://img.shields.io/badge/mPLUG_Owl-https://github.com/XPLUG/mPLUGOwl-blue'></a>
<a href='https://github.com/InternLM/InternLM-XComposer'><img src='https://img.shields.io/badge/InternLM_XComposer-https://github.com/InternLM/InternLMXComposer-blue'></a>
<a href='https://github.com/shikras/shikra'><img src='https://img.shields.io/badge/Shikra-https://github.com/shikras/shikra-blue'></a>
<a href='https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX'><img src='https://img.shields.io/badge/SPHINX-https://github.com/AlphaVLLM/LLaMA2Accessory/tree/main/SPHINX-blue'></a>
<a href='https://huggingface.co/listen2you002/ChartLlama-13b'><img src='https://img.shields.io/badge/ChartLLaMA-https://huggingface.co/listen2you002/ChartLlama13b-blue'></a>
<a href='https://github.com/UniModal4Reasoning/ChartVLM'><img src='https://img.shields.io/badge/ChartVLM-https://github.com/UniModal4Reasoning/ChartVLM-blue'></a>
<a href='https://github.com/THUDM/CogVLM'><img src='https://img.shields.io/badge/CogVLM-https://github.com/THUDM/CogVLM-blue'></a>
<a href='https://github.com/InternLM/InternLM-XComposer'><img src='https://img.shields.io/badge/InternLM_XComposer_v2-https://github.com/InternLM/InternLM_XComposer-blue'></a>
<a href='https://mini-gemini.github.io/'><img src='https://img.shields.io/badge/Mini_Gemini-https://mini_gemini.github.io-blue'></a>
<a href='https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl1.5'><img src='https://img.shields.io/badge/DocOwl-https://github.com/X PLUG/mPLUG DocOwl/tree/main/DocOwl1.5-blue'></a>
<a href='https://github.com/LingyvKong/OneChart'><img src='https://img.shields.io/badge/OneChart-https://github.com/LingyvKong/OneChart-blue'></a>


## Inference
1. Complete the basic environment setups
2. Set prompt style for both *Acc+* and *NQA* tasks in `./Repos/utils.py`
3. Modify the default path of `CKPT_PATH` in `./Repos/{MODEL_NAME}/infer.py`
4. Reimplement the `load_model` and `model_gen` functions
5. The results are saved in `./Result/raw/{MODEL_NAME}.jsonl` by default
6. Prompt LLMs in `./Stat/gpt_filter.py` to extract number values in NQA task
7. Set the parameters in `./Stat/stat_all_metric.py` and the statistical results are saved in `./Stat/Paper_Table`

## Ranking

![ChartBench Pipeline.](./asset/rank.png)

## Citation

```bib
@article{ChartBench,
    title={ChartBench: A Benchmark for Complex Visual Reasoning in Charts},
    author={Zhengzhuo Xu and Sinan Du and Yiyan Qi and Chengjin Xu and Chun Yuan and Jian Guo},
    journal={ArXiv},
    year={2023},
    volume={abs/2312.15915},
    url={https://api.semanticscholar.org/CorpusID:266550948}
}
```