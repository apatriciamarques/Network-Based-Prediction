import subprocess
import time

scripts = [
    "pipeline_algebra.py",
    "pipeline_algebra_cache.py"
]

for script in scripts:
    print(f"Running {script}...")
    start_time = time.time()
    subprocess.run(["python", script], check=True)
    elapsed = time.time() - start_time
    print(f"{script} runtime: {elapsed:.4f} seconds\n")