"""
app.py - Streamlit Interactive Dashboard for Autonomous Coding Agent
"""

import streamlit as st
import time
from agent import AutonomousCodingAgent, LLMClient
from sandbox import DockerSandboxRunner

st.set_page_config(page_title="Autonomous Coding Agent", layout="wide")

st.title("🤖 Autonomous Coding Agent with Docker Sandbox")
st.markdown("Enter a coding specification in plain English. The agent will iteratively **Generate → Execute → Observe → Repair** inside an isolated container.")

with st.sidebar:
    st.header("⚙️ Configuration")
    model_name = st.selectbox("LLM Model", ["gemini-3.8-flash"], index=0)
    max_retries = st.slider("Max Retries (Self-Repair)", min_value=1, max_value=5, value=3)
    mem_limit = st.selectbox("Docker Memory Cap", ["256m", "512m", "1g"], index=0)

# Dynamic Task Input (No Hardcoding)
user_prompt = st.text_area(
    "Task Specification:",
    placeholder="e.g. Write a script to calculate all prime numbers up to 100 and display them in reverse order.",
    height=100
)

if st.button("🚀 Run Autonomous Agent", type="primary"):
    if not user_prompt.strip():
        st.warning("Please enter a task description first.")
    else:
        status_box = st.empty()
        status_box.info("Initializing Agent and Sandbox...")

        llm = LLMClient(model_name=model_name)
        sandbox = DockerSandboxRunner(mem_limit=mem_limit, network_disabled=True)
        agent = AutonomousCodingAgent(llm_client=llm, sandbox=sandbox, max_retries=max_retries, verbose=False)

        with st.spinner("Agent running iterative execution loop..."):
            result_state = agent.run(user_prompt)

        status_box.empty()

        # Display Final Verdict
        if result_state.is_success:
            st.success(f" Task Solved Successfully in {len(result_state.iterations)} attempt(s) ({result_state.total_duration:.2f}s)!")
        else:
            st.error(f"❌ Task Failed after {max_retries} attempts.")

        # Show telemetry metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Attempts Used", len(result_state.iterations))
        col2.metric("Total Tokens", result_state.total_tokens)
        col3.metric("Duration", f"{result_state.total_duration:.2f}s")

        st.divider()

        # Step-by-Step Iteration Timeline
        st.subheader("Inspection: GEOR Iterations")
        for log in result_state.iterations:
            with st.expander(f"Attempt {log.attempt} — Exit Code: {log.exit_code} ({log.duration:.2f}s)", expanded=(log.attempt == len(result_state.iterations))):
                st.markdown("**Generated Python Code:**")
                st.code(log.code_generated, language="python")
                
                if log.stdout.strip():
                    st.markdown("**STDOUT:**")
                    st.code(log.stdout, language="bash")
                
                if log.stderr.strip():
                    st.markdown("**STDERR / Failure Traceback:**")
                    st.code(log.stderr, language="bash")

        if result_state.is_success:
            st.subheader("Final Output")
            st.code(result_state.final_output)