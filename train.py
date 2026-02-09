import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# --- CONFIGURATION ---
# We keep the same 1.5B model (It's efficient and you likely already have it).
# However, we will train it to be a "Standard Support Agent" instead of a "Thinker".
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR = "./my_support_bot"

# --- 1. SETUP MODEL FOR CPU ---
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32, 
    device_map="cpu",
    low_cpu_mem_usage=True
)

# --- 2. SETUP LORA (The Efficiency Layer) ---
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,           # Increased rank slightly for better language quality
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] # Train more layers for style
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 3. PREPARE BITEXT DATASET ---
print("Loading Bitext Customer Support Dataset...")
# This dataset is large (27k rows). We'll take 200 examples for a quick laptop run.
# Change split="train" to use the whole thing (will take hours on CPU).
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train[:200]")

def format_prompts(examples):
    # Standard Professional Chat Format (No <think> tags needed here)
    texts = []
    for instruction, response in zip(examples['instruction'], examples['response']):
        # We use a standard "User/Assistant" format
        prompt = f"User: {instruction}\nAssistant: {response}{tokenizer.eos_token}"
        texts.append(prompt)
    
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256 # Short interactions are faster to train
    )

print("Formatting data...")
tokenized_datasets = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)

# --- 4. TRAIN ---
print("Starting Training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=1,
    use_cpu=True,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# --- 5. SAVE ---
print("Saving your Professional Support Bot...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done! Run 'run_support.py' to test it.")