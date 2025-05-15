# TinfoilHat Chatbot

TinfoilHat is the world's most deranged and mad chatbot. Designed to be an all-in-one source of all the crazy conspiracy theories imaginable, TinfoilHat is here to provide you with the most absurd and outlandish responses to your questions. Whether you're looking for a wild theory about aliens, government cover-ups, or secret societies, TinfoilHat has got you covered.

### Set-up Environment

To set up TinfoilHat, you need to create a virtual environment and install the required dependencies. You can do this by running the following commands:

```bash
python3 -m venv  .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sensitive data is stored in a `.env` file. The format is the following:

```bash
BASE_API="..."
BASE_URL="..."
MODEL_NAME="..."
EMBEDDING_MODEL="..."
API_KEY="..."
```

(Aici mi-am facut pentru mine. Eu am ollama si llama3.2 (2gb), asa ca va trebui modificat pt. LM Studio. Sunt sanse sa trebuiasca implementate separat si alt embedder. Am decis pana la urma sa testez pe un model mai mare si sa folosesc ollama pt. conveninta. Plus, din cate am vazut, desi nu e posibil sa facem direct finetune in astea 2 aplicatii, e posibil sa facem importam in ele modele la care le-am dat finetune (courtesy of ChatGPT))


### Set-up Vector Store

TinfoilHat uses a verity of documents as the source of its outlandish knowledge. Those documents are located in the `conspiracy_documents` folder. Feel free to add or remove any document. Be wary, they still should contain insanity!

Afterwards, you can run the following command to set up the vector store:

```bash
cd vectordb
python3 vector_db.py
```