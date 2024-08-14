export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.99

# conda activate onechart

python infer_chartqa.py 2>&1 | tee -a ./chartqa.log