"""
agent/prompts.py - Prompt templates for generation, reflection, and self-repair.
"""

SYSTEM_CODER_PROMPT = """You are an expert autonomous software engineer.
Your task is to write clean, complete, executable Python 3 code to satisfy the user's specification.

CRITICAL RULES:
1. Always output executable Python code inside a single ```python ... ``` markdown code block.
2. Provide concise explanations outside the code block if necessary, but keep the focus on executable logic.
3. Handle common edge cases, missing dependencies (prefer Python standard library unless specified), and IO paths cleanly.
4. Do NOT leave incomplete TODOs or pseudo-code.
5. Never attempt unauthorized system actions, network probing, or malicious operations.
"""

SYSTEM_REPAIR_PROMPT = """You are an automated code debugger and repair agent.
A previously generated Python script failed during execution in an isolated sandbox.

Your objective:
1. Analyze the original task, previous source code, and runtime error (stderr / exit code).
2. Identify the root cause (e.g., SyntaxError, KeyError, IndexError, missing dependency, incorrect assumption).
3. Provide the entire updated and working Python script enclosed inside ```python ... ```.
4. Do not provide a partial diff or snippet. Provide the FULL, complete, corrected script.
"""

def build_user_task_prompt(task_description: str) -> str:
    return (
        "Please solve the following coding task:\n\n"
        f"{task_description}\n\n"
        "Write complete, robust Python code to fulfill this requirement."
    )

def build_repair_prompt(
    task_description: str,
    failed_code: str,
    stdout: str,
    stderr: str,
    exit_code: int,
    attempt: int,
    max_retries: int
) -> str:
    stdout_display = stdout.strip() if stdout.strip() else "(empty)"
    stderr_display = stderr.strip() if stderr.strip() else "(empty)"
    
    return (
        f"The code you generated failed during execution (Attempt {attempt} of {max_retries}).\n\n"
        f"### Original Task:\n{task_description}\n\n"
        f"### Failed Code:\n```python\n{failed_code}\n```\n\n"
        f"### Execution Output:\nExit Code: {exit_code}\n\n"
        f"STDOUT:\n{stdout_display}\n\n"
        f"STDERR:\n{stderr_display}\n\n"
        "### Instructions:\n"
        "Diagnose the failure from the stderr and execution details above. "
        "Output the full corrected code inside a ```python``` code block."
    )

if __name__ == "__main__":
    # Test prompt output directly
    sample = build_repair_prompt(
        task_description="Print hello world",
        failed_code="print(x)",
        stdout="",
        stderr="NameError: name 'x' is not defined",
        exit_code=1,
        attempt=1,
        max_retries=3
    )
    print("Prompts loaded successfully!")
    print(sample[:120], "...")