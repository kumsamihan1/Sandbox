import os
from agent import AutonomousCodingAgent, LLMClient
from sandbox import DockerSandboxRunner

if __name__ == "__main__":
    llm = LLMClient(model_name="gemini-3.8-flash")
    sandbox = DockerSandboxRunner(mem_limit="256m", network_disabled=True)
    agent = AutonomousCodingAgent(llm_client=llm, sandbox=sandbox, max_retries=3)

    task = "Write a python script that calculates the factorial of 10 and prints the result as integer."
    print("Executing task in Docker sandbox...")
    result = agent.run(task)
    
    print("\n--- Summary ---")
    print(result.summary())