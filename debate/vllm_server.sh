vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
--port 8000 \
--host 0.0.0.0 \
--dtype bfloat16 \
--limit-mm-per-prompt image=16,video=0 \
--tensor_parallel_size 8 \
--served-model-name Qwen2.5-VL-72B-Instruct \
--gpu-memory-utilization 0.8