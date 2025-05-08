import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate

model_id = "phi_finetuned"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

template = """Given the following prompt, provide a relevant and concise response:
prompt: {question}
response: """

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

def generate_response(question: str):
    prompt = QA_CHAIN_PROMPT.format(question=question)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output_ids = model.generate(input_ids, max_length=256, num_return_sequences=1)

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

question = "Is the government aware of alien abductions?"

response = generate_response(question)

print(response)
