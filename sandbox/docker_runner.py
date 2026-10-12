"""
sandbox/docker_runner.py - Secure local Docker execution harness.
"""

import os
import tempfile
import time
from typing import Optional
from agent.state import SandboxResult

try:
    import docker
    from docker.errors import ContainerError, ImageNotFound, APIError
except ImportError:
    docker = None


class DockerSandboxRunner:
    """
    Executes Python scripts inside an ephemeral, resource-governed Docker container.
    Satisfies the SandboxRunner protocol expected by agent/orchestrator.py.
    """
    def __init__(
        self,
        image_name: str = "python:3.11-slim",
        mem_limit: str = "256m",
        nano_cpus: int = 1_000_000_000,  # 1.0 CPU
        network_disabled: bool = True     # Prevent outbound malicious requests
    ):
        if docker is None:
            raise RuntimeError("docker SDK not installed. Run: pip install docker")
        
        self.client = docker.from_env()
        self.image_name = image_name
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus
        self.network_disabled = network_disabled

    def run_code(self, code: str, timeout_sec: int = 15) -> SandboxResult:
        t0 = time.time()
        
        # 1. Create a temporary local file to mount into Docker
        with tempfile.TemporaryDirectory() as host_temp_dir:
            host_script_path = os.path.join(host_temp_dir, "script.py")
            with open(host_script_path, "w", encoding="utf-8") as f:
                f.write(code)

            container = None
            try:
                # 2. Run ephemeral container with hard cgroup restrictions
                container = self.client.containers.run(
                    image=self.image_name,
                    command=["python", "-u", "/workspace/script.py"],
                    volumes={
                        host_temp_dir: {"bind": "/workspace", "mode": "ro"}  # Read-only mount
                    },
                    working_dir="/workspace",
                    mem_limit=self.mem_limit,
                    nano_cpus=self.nano_cpus,
                    network_disabled=self.network_disabled,
                    detach=True,
                    stdout=True,
                    stderr=True,
                )

                # 3. Wait for execution with hard timeout enforcement
                try:
                    res = container.wait(timeout=timeout_sec)
                    exit_code = res.get("StatusCode", 0)
                    stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
                    stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
                    duration = time.time() - t0
                    
                    return SandboxResult(
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=exit_code,
                        duration=duration,
                        timed_out=False
                    )

                except Exception as wait_err:
                    # Watchdog triggered: Process exceeded execution quota
                    container.kill()
                    return SandboxResult(
                        stdout="",
                        stderr=f"TimeoutExpired: Execution exceeded hard ceiling of {timeout_sec} seconds.",
                        exit_code=-1,
                        duration=timeout_sec,
                        timed_out=True
                    )

            except Exception as e:
                return SandboxResult(
                    stdout="",
                    stderr=str(e),
                    exit_code=1,
                    duration=time.time() - t0
                )
            finally:
                # 4. Ephemeral cleanup: Always delete container
                if container:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass