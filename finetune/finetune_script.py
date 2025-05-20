import json
from peft import TaskType, get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from datasets import Dataset, logging


# 1. Load and prepare data with system/user/assistant messages separately
data_path = "../custom_datasets/medeea_fine_tuning_prompts.jsonl"
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

formatted_data = []
for entry in raw_data:
    messages = entry["messages"]
    system_ms = next(m["content"] for m in messages if m["role"] == "system")
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
    formatted_data.append({
        "system": system_ms,
        "user": user_msg,
        "assistant": assistant_msg
    })

dataset = Dataset.from_list(formatted_data)


# 2. Load tokenizer and model
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Add your special tokens (if not already present)
special_tokens_dict = {
    "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]
}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))


# 4. Preprocess function with special tokens inserted explicitly
def preprocess(example):
    system_text = example["system"].strip()
    user_text = example["user"].strip()
    assistant_text = example["assistant"].strip()

    # Compose text with special tokens
    full_text = (
        f"<|system|>\n{system_text}\n"
        f"<|user|>\n{user_text}\n"
        f"<|assistant|>\n{assistant_text}"
    )

    context_text = (
        f"<|system|>\n{system_text}\n"
        f"<|user|>\n{user_text}\n"
    )

    full_encoding = tokenizer(full_text, truncation=True, max_length=512)
    context_encoding = tokenizer(context_text, truncation=True, max_length=512)

    labels = full_encoding["input_ids"].copy()
    # Mask the system + user tokens so the loss is only computed on assistant tokens
    labels[:len(context_encoding["input_ids"])] = [-100] * len(context_encoding["input_ids"])

    full_encoding["labels"] = labels

    return full_encoding


logging.set_verbosity_info()
tokenized_dataset = dataset.map(preprocess, batched=False)

# 5. LoRA config (same as before)
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

training_args = TrainingArguments(
    output_dir="../phi-lora-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    bf16=False,
    fp16=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model()
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

print(f"Fine tuned and saved Model to {training_args.output_dir}")
