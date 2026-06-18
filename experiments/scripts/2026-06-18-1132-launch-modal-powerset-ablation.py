import modal
import os
import subprocess
import time
import glob

app = modal.App("cooking-world-ablation")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("default-jre", "curl", "git", "build-essential", "ninja-build")
    .pip_install("uv")
    .add_local_file("pyproject.toml", "/root/project/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/root/project/uv.lock", copy=True)
    .run_commands("cd /root/project && uv sync --frozen --all-extras")
    .add_local_dir(".", remote_path="/root/project", ignore=["**/.venv/**", "**/__pycache__/**", "**/.git/**"])
)

results_volume = modal.Volume.from_name("cooking-world-results", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100", 
    timeout=43200, 
    volumes={"/root/results": results_volume}, 
    secrets=[modal.Secret.from_name("my-huggingface-secret")]
)
def run_powerset_ablation():
    import subprocess
    import time
    import os
    
    os.chdir("/root/project")
    
    vllm_wrapper = """
import sys
import runpy

try:
    from prometheus_fastapi_instrumentator import routing
    routing.get_route_name = lambda request: "vllm_endpoint"
except ImportError:
    pass

sys.argv = [
    "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen2.5-7B-Instruct-1M", 
    "--port", "8000",
    "--max-model-len", "131072"
]
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
"""

    print("Starting vLLM server in the background...")
    vllm_process = subprocess.Popen(
        ["/root/project/.venv/bin/python", "-c", vllm_wrapper]
    )
    
    print("Waiting for vLLM to initialize...")
    import urllib.request
    while True:
        if vllm_process.poll() is not None:
            raise RuntimeError("vLLM server crashed during initialization!")
        try:
            req = urllib.request.urlopen("http://localhost:8000/v1/models")
            if req.getcode() == 200:
                print("vLLM is up and ready!")
                break
        except Exception:
            pass
        time.sleep(5)
    
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "sk-fake"
    env["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
    
    print("Starting ablation script...")
    
    subprocess.run(
        [
            "/root/project/.venv/bin/python", "experiments/scripts/2026-06-17-1312-powerset-ablation.py",
            "--model", "openai/Qwen/Qwen2.5-7B-Instruct-1M",
            "--max-connections", "16",
            "--max-tasks", "8"
        ],
        env=env,
        check=True
    )
    
    vllm_process.terminate()
    
    print("Experiment finished. Saving output files to persistent volume...")
    
    os.system("cp experiments/experiment-output/*.json /root/results/")
    
    results_volume.commit()
    print("Files committed to Modal Volume!")


@app.local_entrypoint()
def main():
    print("Spawning ablation job to run in the cloud...")
    
    run_powerset_ablation.remote()
    
    print("\n✅ Job is finished! You can download all the JSON results directly from the persistent volume using:")
    print("  modal volume get cooking-world-results * ./experiments/experiment-output/")

