from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_name = "microsoft/Phi-3.5-mini-instruct"
lora_model_path = "../phi-lora-finetuned"
merged_model_path = "../phi-lora-merged"

# Load tokenizer and add special tokens (same as during training)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
special_tokens = ["<|system|>", "<|user|>", "<|assistant|>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Resize token embeddings to match tokenizer length
model.resize_token_embeddings(len(tokenizer))

# Now load the LoRA adapter on the resized model
model = PeftModel.from_pretrained(model, lora_model_path)

# Merge adapter weights into base model
model = model.merge_and_unload()

# Save the merged full model to disk
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
