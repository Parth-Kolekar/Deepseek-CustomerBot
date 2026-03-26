# 🤖 DeepSeek Customer Support Bot

A lightweight, locally-runnable customer support AI agent. This project fine-tunes the `DeepSeek-R1-Distill-Qwen-1.5B` model using LoRA (Low-Rank Adaptation) to act as a professional, polite, and helpful customer support representative. 

The inference script is specifically optimized to run efficiently on CPUs and features real-time text streaming.

## ✨ Features
* **Professional Persona:** Fine-tuned to de-escalate frustrated customers and provide concise, helpful answers.
* **CPU Optimized:** Uses `transformers` and `peft` with low CPU memory usage mapping to run locally without an expensive GPU.
* **Real-time Streaming:** Implements `TextStreamer` for a fluid, word-by-word typing effect in the terminal.
* **Cloud-Hosted Adapter:** Automatically downloads the fine-tuned LoRA weights directly from the Hugging Face Hub.

---

## ☁️ How to Use Directly from Hugging Face (No Git Clone Required)

If you just want to use the fine-tuned model in your own Python application, you don't need to download this repository. You can load it directly from the Hugging Face Hub using the `transformers` and `peft` libraries.

### 1. Install Dependencies
```bash
pip install torch transformers peft accelerate


### 2. Load and Run the Model
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Define models
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTER_MODEL = "parthwolverine81/deepseek-customer-bot-cpu-finetuned"

# 2. Load Base Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

# 3. Load Adapter from Hugging Face and Merge
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
model = model.merge_and_unload()

# 4. Generate a Response
prompt = "User: My order is delayed and I'm very frustrated!\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("Assistant:")[-1].strip())
```

---

## 💻 Local Setup & Development

If you want to run the provided terminal chat application or retrain the model, follow these steps:

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

### 2. Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/deepseek-customerbot.git
cd deepseek-customerbot
pip install -r requirements.txt
```

### 3. Running the Terminal Bot
Launch the interactive terminal chat application, which includes real-time streaming:

```bash
python run.py
```

You can then chat with the bot directly in your terminal:

```plaintext
--- Optimized Professional Support Bot (CPU) ---
Type 'exit' to quit.

Customer: What are your business hours?
Agent: Our business hours are Monday through Friday, 9:00 AM to 5:00 PM EST. How else can I assist you today?
```

---

## 🛠️ Project Structure
* `train.py`: Script used to fine-tune the model on customer support data.
* `run.py`: Optimized CPU inference script with interactive terminal streaming.
* `requirements.txt`: Python dependencies.
* `my_support_bot/`: Local directory containing adapter weights, tokenizer configs, and chat templates.

---

## 🧠 Model Details
* **Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
* **Fine-tuned Adapter:** `parthwolverine81/deepseek-customer-bot-cpu-finetuned`
* **Training Method:** PEFT / LoRA
``` the markdown formatting?
