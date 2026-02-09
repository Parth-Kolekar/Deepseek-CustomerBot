# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# ADAPTER_FOLDER = "./my_support_bot"

# print("Loading model...")
# model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     torch_dtype=torch.float32,
#     device_map="cpu",
#     low_cpu_mem_usage=True
# )
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# print("Loading Adapter...")
# model = PeftModel.from_pretrained(model, ADAPTER_FOLDER)
# model = model.merge_and_unload()

# print("\n--- Professional Support Bot (CPU) ---")
# print("Type 'exit' to quit.\n")

# while True:
#     user_input = input("Customer: ")
#     if user_input.lower() in ["exit", "quit"]: break

#     # Use the same format as training
#     prompt = f"User: {user_input}\nAssistant:"
    
#     inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

#     print("Agent is typing...", end="", flush=True)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=150,
#             temperature=0.3, # Low temperature = More professional/consistent
#             do_sample=True,
#             repetition_penalty=1.2 # Prevents repeating phrases
#         )
    
#     print("\r", end="")
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # Extract only the assistant's part
#     if "Assistant:" in response:
#         response = response.split("Assistant:")[-1].strip()
    
#     print(f"Agent: {response}\n")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTER_FOLDER = "./my_support_bot"

print("Loading model...")
# Optimization: Dynamic quantization can significantly speed up CPU inference
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading Adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_FOLDER)
model = model.merge_and_unload()

# --- OPTIMIZATION: Setup Streaming ---
# This prints the response token-by-token instead of waiting for the end
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("\n--- Optimized Professional Support Bot (CPU) ---")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Customer: ")
    if user_input.lower() in ["exit", "quit"]: break

    # The prompt format helps suppress the <think> tags you saw in your screenshot
    prompt = f"User: {user_input}\nAssistant:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    print("Agent: ", end="", flush=True)
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3, 
            do_sample=True,
            streamer=streamer, # Enables the word-by-word display
            pad_token_id=tokenizer.eos_token_id
        )
    print("\n")