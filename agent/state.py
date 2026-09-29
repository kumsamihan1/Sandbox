"""
agent/state.py - Dataclasses representing execution attempts and the agent run state.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time

@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    duration: float
    timed_out: bool = False

@dataclass
class IterationLog:
    attempt: int
    code_generated: str
    stdout: str
    stderr: str
    exit_code: int
    duration: float
    prompt_tokens: int = 0
    completion_tokens: int = 0

@dataclass
class AgentState:
    task: str
    max_retries: int
    is_success: bool = False
    final_code: Optional[str] = None
    final_output: Optional[str] = None
    final_error: Optional[str] = None
    total_duration: float = 0.0
    total_tokens: int = 0
    iterations: List[IterationLog] = field(default_factory=list)

    def add_iteration(
        self,
        attempt: int,
        code: str,
        sandbox_res: SandboxResult,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ):
        log = IterationLog(
            attempt=attempt,
            code_generated=code,
            stdout=sandbox_res.stdout,
            stderr=sandbox_res.stderr,
            exit_code=sandbox_res.exit_code,
            duration=sandbox_res.duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self.iterations.append(log)
        self.total_tokens += (prompt_tokens + completion_tokens)

    def summary(self) -> Dict[str, Any]:
        return {
            "success": self.is_success,
            "attempts": len(self.iterations),
            "total_tokens": self.total_tokens,
            "total_duration_sec": round(self.total_duration, 2),
            "final_code": self.final_code,
            "final_output": self.final_output,
            "final_error": self.final_error,
        }