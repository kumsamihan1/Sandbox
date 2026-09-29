"""
agent/orchestrator.py - State machine loop implementing Generate -> Execute -> Observe -> Repair.
"""

import time
from typing import Protocol, runtime_checkable
from .prompts import (
    SYSTEM_CODER_PROMPT,
    SYSTEM_REPAIR_PROMPT,
    build_user_task_prompt,
    build_repair_prompt,
)
from .state import AgentState, SandboxResult
from .llm import LLMClient

@runtime_checkable
class SandboxRunner(Protocol):
    """Protocol that Member 2's sandbox (Docker/E2B) must satisfy."""
    def run_code(self, code: str, timeout_sec: int = 15) -> SandboxResult:
        ...

class AutonomousCodingAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        sandbox: SandboxRunner,
        max_retries: int = 3,
        verbose: bool = True
    ):
        self.llm = llm_client
        self.sandbox = sandbox
        self.max_retries = max_retries
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[AutonomousAgent] {msg}")

    def run(self, task: str) -> AgentState:
        """
        Runs the self-correcting autonomous coding loop.
        """
        state = AgentState(task=task, max_retries=self.max_retries)
        start_time = time.time()

        self._log(f"Received new task: {task[:70]}...")
        
        # 1. Initial Generation
        messages = [
            {"role": "system", "content": SYSTEM_CODER_PROMPT},
            {"role": "user", "content": build_user_task_prompt(task)},
        ]

        current_code = ""
        
        for attempt in range(1, self.max_retries + 1):
            self._log(f"--- Iteration {attempt}/{self.max_retries} ---")
            
            # Step A: LLM Code Generation / Repair
            self._log("Querying LLM for code solution...")
            raw_response, p_tok, c_tok = self.llm.query(messages)
            current_code = self.llm.extract_python_code(raw_response)

            if not current_code:
                self._log("Warning: No code could be extracted from LLM response.")
                current_code = raw_response

            # Step B: Sandbox Execution
            self._log("Dispatching code to sandbox execution environment...")
            sandbox_res: SandboxResult = self.sandbox.run_code(current_code)
            
            # Step C: Record Iteration State
            state.add_iteration(
                attempt=attempt,
                code=current_code,
                sandbox_res=sandbox_res,
                prompt_tokens=p_tok,
                completion_tokens=c_tok
            )

            # Step D: Observe & Evaluate Result
            if sandbox_res.exit_code == 0 and not sandbox_res.timed_out:
                self._log(f"Success on attempt {attempt}! Exit code 0.")
                state.is_success = True
                state.final_code = current_code
                state.final_output = sandbox_res.stdout
                break
            else:
                self._log(f"Attempt {attempt} failed with exit code {sandbox_res.exit_code}.")
                if sandbox_res.timed_out:
                    self._log("Reason: Execution timed out.")

                # If attempts exhausted, break
                if attempt == self.max_retries:
                    self._log("Max retries reached. Self-repair loop terminated.")
                    state.final_code = current_code
                    state.final_error = sandbox_res.stderr or "Execution timed out / failed"
                    break

                # Step E: Construct Repair Loop Context
                repair_prompt = build_repair_prompt(
                    task_description=task,
                    failed_code=current_code,
                    stdout=sandbox_res.stdout,
                    stderr=sandbox_res.stderr,
                    exit_code=sandbox_res.exit_code,
                    attempt=attempt,
                    max_retries=self.max_retries,
                )

                messages = [
                    {"role": "system", "content": SYSTEM_REPAIR_PROMPT},
                    {"role": "user", "content": repair_prompt}
                ]

        state.total_duration = time.time() - start_time
        return state