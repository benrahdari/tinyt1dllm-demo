# Tiny T1D LLM Demo
This script replicates the experiment described in the paper "Local by Design: Tiny AI Agents for Hyper-Personalized Longitudinal Health Support at Home". It demonstrates how a small, local Large Language Model (LLM) can analyze time-series data from a Continuous Glucose Monitor (CGM) and generate a human-readable summary of the trends.

## Core Task:
1. Load CGM data from the provided local XML file (559-ws-testing.xml).
2. Use a compact, quantized LLM (ex. microsoft-phi2) that can run efficiently on a CPU.
3. Craft a prompt instructing the model to summarize glucose trends.
4. Generate the summary and measure performance (execution time and RAM usage).

## Requirements:
You will need to install the following libraries to run this script:
```
pip install transformers torch psutil
```
