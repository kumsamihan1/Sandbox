"""
evals/runner.py - Automated benchmark evaluation harness.
"""

import json
import os
import time
from pathlib import Path
from agent import AutonomousCodingAgent, LLMClient, SandboxResult

# Reuse the local test runner until Docker runner is merged
from test_local import LocalSubprocessSandbox

def run_evaluation_suite(dataset_path: str, max_retries: int = 3):
    with open(dataset_path, "r") as f:
        tasks = json.load(f)

    sandbox = LocalSubprocessSandbox()
    llm = LLMClient(model_name="gpt-4o-mini")
    agent = AutonomousCodingAgent(llm_client=llm, sandbox=sandbox, max_retries=max_retries, verbose=False)

    total_tasks = len(tasks)
    zero_shot_passes = 0
    repaired_passes = 0
    total_failures = 0
    total_tokens_consumed = 0

    results = []

    print(f"\n=======================================================")
    print(f"🚀 RUNNING BENCHMARK EVALUATION ({total_tasks} Tasks, Max Retries: {max_retries})")
    print(f"=======================================================\n")

    for idx, item in enumerate(tasks, start=1):
        task_id = item["id"]
        name = item["name"]
        prompt = item["prompt"]
        expected = item.get("expected_output_contains", "").strip()

        print(f"[{idx}/{total_tasks}] Evaluating: {name} ({task_id})...", end=" ", flush=True)
        
        t0 = time.time()
        state = agent.run(prompt)
        elapsed = round(time.time() - t0, 2)
        total_tokens_consumed += state.total_tokens

        # Check correctness against assertion
        stdout = (state.final_output or "").strip()
        matched = expected in stdout if expected else state.is_success
        attempts_used = len(state.iterations)

        if state.is_success and matched:
            if attempts_used == 1:
                zero_shot_passes += 1
                status = "PASS (Zero-Shot)"
            else:
                repaired_passes += 1
                status = f"PASS (Repaired on attempt {attempts_used})"
        else:
            total_failures += 1
            status = "FAIL"

        print(f"{status} [{elapsed}s, {state.total_tokens} tokens]")

        results.append({
            "id": task_id,
            "name": name,
            "status": status,
            "attempts": attempts_used,
            "tokens": state.total_tokens,
            "duration_sec": elapsed,
            "final_output": stdout[:100]
        })

    # Summary Statistics
    total_passed = zero_shot_passes + repaired_passes
    pass_rate = (total_passed / total_tasks) * 100
    zero_shot_rate = (zero_shot_passes / total_tasks) * 100
    repair_lift = ((repaired_passes) / total_tasks) * 100

    print("\n================ BENCHMARK REPORT ================")
    print(f"Total Tasks        : {total_tasks}")
    print(f"Zero-Shot Accuracy : {zero_shot_rate:.1f}% ({zero_shot_passes}/{total_tasks})")
    print(f"Self-Repair Boost  : +{repair_lift:.1f}% ({repaired_passes} recovered)")
    print(f"Final Pass Rate    : {pass_rate:.1f}% ({total_passed}/{total_tasks})")
    print(f"Total Tokens Used  : {total_tokens_consumed}")
    print("==================================================")

    # Save to JSON for report/charts
    output_file = Path("evals/benchmark_results.json")
    with open(output_file, "w") as out:
        json.dump({
            "metrics": {
                "total_tasks": total_tasks,
                "pass_rate": pass_rate,
                "zero_shot_rate": zero_shot_rate,
                "repair_recovery_rate": repair_lift,
                "total_tokens": total_tokens_consumed
            },
            "task_runs": results
        }, out, indent=2)
    print(f"\nSaved detailed metrics to {output_file.resolve()}")

if __name__ == "__main__":
    run_evaluation_suite("evals/tasks.json")