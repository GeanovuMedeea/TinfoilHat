# 🤖 TinfoilHat Chatbot

> “The truth is out there. And it’s **completely insane.” 🛸📡🧢

*TinfoilHat* is the world’s most deranged, paranoid, and entertaining chatbot. It uses Retrieval-Augmented Generation (RAG) with a fine-tuned LLM to simulate a conspiracy theorist who trusts everything except the truth. Whether you're curious about alien pyramids on Mars, 5G brain rays, or reptilian time travelers, TinfoilHat is ready to theorize.

---

## 🧠 Project Overview

TinfoilHat is an LLM-powered chatbot that simulates the persona of a deeply paranoid AI. It uses vector search to retrieve conspiracy-based document chunks and combines them with a fine-tuned LLM response generator.

- RAG-powered conspiracy theorizing
- LoRA-finetuned model on top of Phi-3.0-mini-instruct. Other option is Phi-4 until the finetuned model will reach a satisfactory level in output generation and performance.
- And it's absurd, sarcastic, and always suspicious! Great combo.

---

## 📊 Architecture Diagram

text
User Query
   │
   ▼
Embed Query (OllamaEmbedder)
   │
   ▼
Chroma Vector DB (FAISS)
   │
Retrieve Relevant Conspiracy Chunks
   ▼
Construct Prompt (TinfoilHat persona)
   ▼
LLM Response via Ollama or LM Studio
   ▼
Return Hilariously Paranoid Answer


---

## 🧰 Technologies Used

| Component     | Tech                        |
|---------------|-----------------------------|
| LLM           | Phi-3 (LoRA) or Phi-4 |
| Embeddings    | text-embedding-nomic-embed-text-v1.5 |
| Vector DB     | Chroma              |
| Embedding/RAG | LangChain + Ollama + LangChain-Community |
| Toxicity      | Detoxify |
| Finetuning    | LoRA (PEFT)         |
| Interface     | Gradio                      |
| Evaluation    | DeepEval (planned)          |

---

## 🛠 Setup Instructions

### 1. Create Virtual Environment

bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


---

### 2. Configure .env

bash
BASE_API="..."
BASE_URL="..."
MODEL_NAME="..."
EMBEDDING_MODEL="..."
API_KEY="..."


---

## 📁 Folder Structure

bash
project_root/
├── conspiracy_documents/      # Raw PDF conspiracy texts
├── vectordb/                  # Builds vector index
│   └── vector_db.py
├── datasets/                  # Fine-tuning datasets
├── app.py                     # Main Gradio app
├── embedder.py                # OllamaEmbedder class
├── model.py                   # RAG logic
├── .env
└── requirements.txt



## 🧩 Prompt Engineering

TinfoilHat is more than a chatbot — it’s a persona. It never believes official stories and always finds the most outrageous angle.

> “You are a conspiracy theorist who believes in everything but the truth and especially in the official narratives. You come up with some of the wildest ideas known to man. You are incredibly persuasive...”

## Key behaviors:

- Slight sarcasm 🤨
- Heavy emoji use 🛸👁‍🗨📡
- Capitalized emphasis on THE MOST OUTRAGEOUS CLAIMS 😱
- Refuses to believe anything mainstream

## 🛠 Example Prompts

txt
User: What caused the 2003 blackout?

TinfoilHat: Oh you mean the one they BLAMED on a tree branch? 🌳😂 Yeah right. It was obviously an EXPERIMENT in mass EMP deployment by secret government drones 🛸🐦 running on alien tech recovered in 1947. But sure, let’s blame the branch.


---

# 🔍 RAG Pipeline

1. PDF Injestion

- PDFs from conspiracy_documents/ are chunked into ~500-token chunks and embedded using OllamaEmbedder.

2. Vector Store Creation

- Chunks are stored in a persistent Chroma DB (chroma/) using FAISS.

3. Query Handling

- User query → embedded via Ollama
- Top-k chunks retrieved from Chroma
- Prompt is generated with conspiracy persona
- LLM generates answer (via Ollama or custom model)

4. Response

- Response is returned, optionally with source chunks for debugging.

---

# 🧪 Evaluation Strategy

| Metric             | Purpose                                       |
|--------------------|-----------------------------------------------|
| ✅ Faithfulness     | Is the conspiracy sourced from retrieved docs? |
| 🎯 Relevance       | Does the context match the query?             |
| 🧠 Tone Match      | Custom test for “paranoid/deranged” language  |
| 🧢 Outrage Score   | Measures the insanity level    |


---

# 🧬 Fine-Tuning Logic

- Fine-tuned microsoft/phi-1_5 with conspiracy Q&A pairs via LoRA (PEFT)

- Targeted q_proj, k_proj, v_proj, o_proj

- Used JSONL format with structured system, user, assistant roles

- Stored outputs in: datasets/medeea_fine_tuning_prompts.jsonl

To launch training:

bash
python3 train_phi_lora.py  # or integrated inside script


---

# ▶ How to Use (CLI / Chatbot)

bash
python3 app.py


Launches a Gradio chat interface:

> TinFoil Hat 📶🛜📡
> 
> "Talk to the world's most demented AI!"

Ask it:

- “What happened at Area 51?”

- “Is the Moon real?”

- “Was JFK killed by time travelers?”

- “Are vaccines part of a control grid?”

---

# 🧪 Example RAG Prompts (with Source Context)

python
prompts = [
    "Is the Moon artificial?",
    "Are crisis actors used during disasters?",
    "Is there ancient tech buried in Antarctica?"
]


Results are stored in:

json
datasets/bia_testing_prompts.json


# 🛡 Content Policy Safeguards

While the chatbot simulates extreme conspiracies, documents are curated to avoid inciting harm, hate, or violence. This is absurdity for entertainment — not a tool to spread real misinformation.

# 🔮 Future Improvements

- Streamlit Web UI 🖥  
- Multi-modal conspiracy generation (images + text)  
- Automated ingestion from forums & archives (e.g. Reddit, 4chan, FOIA dumps)  
- TinfoilHat ToneScore (evaluates how insane responses sound)  
- RAG Chain Inspector (debug why this chunk was chosen)

# 🎬 Final Note

The truth is classified.  
TinfoilHat is here to speculate wildly and entertain furiously.  
Stay paranoid. 🧢