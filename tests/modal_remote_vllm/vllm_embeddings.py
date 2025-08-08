#!/usr/bin/env python3
"""
Serve embeddings via Arctic-Inference + vLLM in OpenAI-compatible mode on Modal.
Endpoint:  /v1/embeddings

OPTIMIZED FOR A10G GPU:
- max_num_seqs: 256 (increased from 192)
- max_num_batched_tokens: 98304 (doubled from 49152) 
- Modal concurrency: 256 (increased from 128)
- Disabled request logging for reduced overhead
"""

import modal

# ------------ 1. Image ------------
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "python3-pip", "wget", "curl")
    .pip_install(
        "git+https://github.com/snowflakedb/ArcticInference.git",
        "vllm==0.9.2",
        "huggingface_hub[hf_transfer]==0.33.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ------------ 2. Model ------------
MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v1.5"

# ------------ 3. Volumes ------------
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ------------ 4. Cold-start tuning ------------
FAST_BOOT = True

# ------------ 5. Modal App ------------
app = modal.App("arctic-embeddings")

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"A10G:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=256)  # Increased to match max_num_seqs
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--tensor-parallel-size", "1",
        "--max-model-len", "512",
        "--max-num-batched-tokens", "98304",  # Doubled for higher throughput
        "--max-num-seqs", "256",  # Increased from 192 to handle more concurrent requests
        "--gpu-memory-utilization", "0.95",
        "--dtype", "float16",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--disable-log-requests",  # Reduce logging overhead
        "--max-seq-len-to-capture", "512",  # Match max model length
    ]
    print("Launching Arctic-Inference:", " ".join(cmd))
    subprocess.Popen(cmd)


# ------------ 6. Quick Health Check ------------
@app.local_entrypoint()
def entry():
    url = serve.get_web_url()
    print("url:", url)
