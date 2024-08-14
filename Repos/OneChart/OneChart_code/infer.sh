export CUDA_VISIBLE_DEVICES=5
export FLAGS_fraction_of_gpu_memory_to_use=0.99

# conda activate onechart

python infer.py 2>&1 | tee -a ./chartbench.log