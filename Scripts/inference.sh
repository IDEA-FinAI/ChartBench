# ! BLIP2
conda activate xzz
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/BLIP2
CUDA_VISIBLE_DEVICES=0 python run.py

# ! ChartLLaMA
conda activate dsn
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/ChartLLaMA
CUDA_VISIBLE_DEVICES=1 python run.py

# ! CogVLM
conda activate xzz_2.0
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/CogVLM
CUDA_VISIBLE_DEVICES=2 python run.py

# ! InstructBLIP
conda activate xzz_2.0
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/InstructBLIP
CUDA_VISIBLE_DEVICES=3 python run.py

# ! InternLM-XComposer
conda activate xzz_2.0
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/InternLM-XComposer
CUDA_VISIBLE_DEVICES=4 python run.py

# ! LLaVA
conda activate llava
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/LLaVA
CUDA_VISIBLE_DEVICES=5 python run.py

# ! MiniGPT-4
conda activate minigptv
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/MiniGPT-4
CUDA_VISIBLE_DEVICES=6 python run.py

# ! mPLUG-Owl
conda activate mplug_owl
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/mPLUG-Owl
CUDA_VISIBLE_DEVICES=1 python run.py

# ! Qwen-VL-Chat
conda activate xzz_2.0
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/Qwen-VL-Chat
CUDA_VISIBLE_DEVICES=2 python run.py

# ! shikra
conda activate dsn
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/shikra
CUDA_VISIBLE_DEVICES=1 python run.py

# ! SPHINX
conda activate sphinx
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/SPHINX
CUDA_VISIBLE_DEVICES=5,6 python run.py

# ! VisualGLM-6B
conda activate minigptv
cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/VisualGLM-6B
CUDA_VISIBLE_DEVICES=6 python run.py


# =========================
# ! Fuyu-8B
# conda activate xzz
# cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/Fuyu-8B
# CUDA_VISIBLE_DEVICES=5 python run.py

# ! LaVIN
# conda activate dsn
# cd /data/FinAi_Mapping_Knowledge/qiyiyan/xzz/ChartBench/Repos/LaVIN
# CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node 1 run.py
