import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate

from vector_db import vectordb

model_id = "phi-lora-finetuned"  # Fine-tuned Phi-1.5 model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Wrap your Phi model with HuggingFacePipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
llm = HuggingFacePipeline(pipeline=generator)

# Custom prompt template: Use retrieved context only as a guide for boosting
template = """You have been trained on the following information. Use this to answer the question as clearly as possible, combining it with your own knowledge. Answer in a paragraph, at least 3 sentences.

{context}

Question: {question}
Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "Was the moon landing a hoax?"

result = qa_chain({"query": question})

print(result["result"])


