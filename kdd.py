# ==============================================================================
# KDD 2025 HealthDay Demo: On-Device LLM for Glucose Trend Analysis
# ==============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import xml.etree.ElementTree as ET
import time
import psutil
import os

def get_process_memory():
    """Helper function to get the current memory usage of the process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # Return in MB

def load_glucose_data_from_xml(filename):
    """
    Parses an XML file to extract glucose level event data.
    It expects the data in the format:
    <patient>
      <glucose_level>
        <event ts="..." value="..."/>
        ...
      </glucose_level>
    </patient>
    """
    print(f"Attempting to load data from '{filename}'...")
    if not os.path.exists(filename):
        print(f"Error: Data file '{filename}' not found.")
        print("Please ensure the XML file is in the same directory as the script.")
        return None
        
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        glucose_readings = []
        # Find all 'event' elements within the 'glucose_level' tag
        for event in root.findall('./glucose_level/event'):
            value = event.get('value')
            if value:
                glucose_readings.append(int(value))
        return glucose_readings
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return None

def run_experiment():
    """
    Main function to run the glucose trend analysis experiment.
    """
    print("======================================================")
    print("Starting On-Device Health AI Experiment...")
    print("======================================================")

    # --- 1. Load the Dataset ---
    # This now loads data from the local XML file you provided.
    print("\n[Step 1/4] Loading patient glucose data from XML file...")
    
    dataset_filename = "559-ws-testing.xml"
    glucose_readings_full = load_glucose_data_from_xml(dataset_filename)

    if not glucose_readings_full:
        print("Failed to load glucose data. Aborting experiment.")
        return

    # To match the paper's experiment, we'll use a slice of 180 readings.
    # If the file contains fewer, we'll use all available readings.
    if len(glucose_readings_full) >= 180:
        glucose_readings = glucose_readings_full[:180]
    else:
        glucose_readings = glucose_readings_full
        
    print(f"Successfully loaded {len(glucose_readings)} glucose readings from '{dataset_filename}'.")
    print("-" * 50)

    # --- 2. Load the On-Device Model ---
    # We select a smaller, distilled model that is optimized for faster
    # inference on standard hardware (like a CPU).
    # NOTE: The first time you run this, it will download the model files,
    # which can be several gigabytes. Subsequent runs will be faster.
    print("\n[Step 2/4] Loading the TinyLLM model...")
    print("This may take a moment and will download model files on first run.")
    mem_before_model = get_process_memory()


    model_name = "microsoft/phi-2"
    # Or select a different model
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # model_name = "openlm-research/open_llama_3b_v2"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # For CPU-only, we explicitly set the torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32 # Use float32 for CPU compatibility
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have a stable internet connection for the first run.")
        return

    mem_after_model = get_process_memory()
    print("Model loaded successfully.")
    print(f"Initial RAM usage: {mem_before_model:.2f} MB")
    print(f"RAM usage after loading model: {mem_after_model:.2f} MB")
    print(f"Model memory footprint: {mem_after_model - mem_before_model:.2f} MB")
    print("-" * 50)

    # --- 3. Construct the Prompt ---
    # This is a critical step. We are creating a "zero-shot" prompt where we
    # provide context and a clear instruction to the model.
    print("\n[Step 3/4] Constructing the prompt for the model...")
    print("Printing clucose_srt: {glucose_str}")

    # Convert the list of readings to a string
    glucose_str = ', '.join(map(str, glucose_readings))

    prompt = f"""You are a helpful and concise AI assistant for diabetes management.
A patient has provided a series of Continuous Glucose Monitoring (CGM) readings (in mg/dL) from the last few hours.
Analyze the following glucose data and provide a brief, easy-to-understand summary of the trend.
Mention the starting and ending levels, the peak level, and describe the overall pattern (e.g., rising, falling, stable, roller-coaster).

Glucose Readings:
{glucose_str}

Summary of the trend:"""

    print("Prompt constructed.")
    print("-" * 50)


    # --- 4. Generate the Summary and Measure Performance ---
    print("\n[Step 4/4] Generating the glucose trend summary...")
    start_time = time.time()

    # Encode the prompt and generate a response from the model
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Generate the response
    with torch.no_grad(): # Inference doesn't require gradients
        output = model.generate(
            input_ids,
            max_new_tokens=150, # Limit the length of the new text
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True, # Use sampling for more natural text
            temperature=0.7,
            top_p=0.9
        )

    end_time = time.time()
    generation_time = end_time - start_time

    # Decode the generated tokens into a string, skipping the prompt part
    response = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
    
    mem_after_generation = get_process_memory()

    print("\n======================================================")
    print("                EXPERIMENT RESULTS                  ")
    print("======================================================")
    print(f"\nAI-Generated Glucose Trend Summary:\n")
    print(response)
    print("\n------------------------------------------------------")
    print(f"Performance Metrics:")
    print(f"   - Generation Time: {generation_time:.2f} seconds")
    print(f"   - Total RAM usage: {mem_after_generation:.2f} MB")
    print("======================================================")


if __name__ == "__main__":
    run_experiment()
