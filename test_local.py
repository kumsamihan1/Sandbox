import subprocess
import time
import os
from agent import AutonomousCodingAgent, LLMClient, SandboxResult

class LocalSubprocessSandbox:
    """Mock sandbox executing locally to verify the agent flow before Docker is connected."""
    def run_code(self, code: str, timeout_sec: int = 10) -> SandboxResult:
        t0 = time.time()
        try:
            proc = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            return SandboxResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                duration=time.time() - t0
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="", 
                stderr="TimeoutExpired: Execution exceeded limit.", 
                exit_code=-1, 
                duration=timeout_sec, 
                timed_out=True
            )

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is available if testing live API calls
    # os.environ["OPENAI_API_KEY"] = "your-key-here"
    
    print("[1/3] Testing package imports...")
    sandbox = LocalSubprocessSandbox()
    print("    Sandbox mock initialized successfully.")

    # Smoke test mock execution
    test_run = sandbox.run_code("print('Hello from Sandbox!')")
    print(f"    Sandbox test stdout: {test_run.stdout.strip()}")
    assert test_run.exit_code == 0

    print("[2/3] Initializing Agent...")
    llm = LLMClient(model_name="gpt-4o-mini")
    agent = AutonomousCodingAgent(llm_client=llm, sandbox=sandbox, max_retries=3)
    print("    AutonomousCodingAgent loaded without errors.")

    print("[3/3] System is ready for live tasks.")
    # To run a real task with your API key set, uncomment:
    # state = agent.run("Generate Python code to print squares of numbers from 1 to 5")
    # print(state.summary())