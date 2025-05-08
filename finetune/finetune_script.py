import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
from datasets import Dataset

# CONFIG
model_id = "microsoft/phi-1_5"
csv_filename = "../datasets/conspiracy_dataset.csv"
output_dir = "../../phi-lora-finetuned"

#LOAD DATASET
df = pd.read_csv(csv_filename)[["Input", "Response"]]
dataset = Dataset.from_pandas(df)

#LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# PREPROCESS FUNCTION
def preprocess(example):
    prompt = f"prompt: {example['Input']}\n{example['Response']}</s>"
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=128)  # Reduced max_length to optimize memory
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#LOAD MODEL
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# LoRA CONFIG
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#TRAINING ARGS
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Fine tuned and saved Model to {output_dir}")
