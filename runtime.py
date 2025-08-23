import subprocess
import time

scripts = [
    "pipeline_algebra.py",
    "pipeline_algebra_changes.py"
]

runtimes = []

for script in scripts:
    print(f"Running {script}...")
    start_time = time.time()
    try:
        # Run the script and capture output
        result = subprocess.run(
            ["python", script],
            check=True,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time
        runtimes.append(f"{script} runtime: {elapsed:.4f} seconds")
        
        # Optional: print script output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")
        runtimes.append(f"{script} failed after {time.time() - start_time:.4f} seconds")

print("\nAll runtimes:")
for r in runtimes:
    print(r)