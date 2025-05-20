from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import SystemMessagePromptTemplate

from filter.toxicity_filter import *
from prompts import system_prompt, rag_user_prompt
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from embedder import ollama_embedder
import os

api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")
base_url = os.getenv("BASE_URL")

print(api_key, model_name, base_url)

llm = ChatOpenAI(
    api_key=api_key,
    model=model_name,
    temperature=0.7,
    base_url=base_url,
)

system_message = SystemMessagePromptTemplate.from_template(system_prompt)
user_message = HumanMessagePromptTemplate.from_template(rag_user_prompt)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        system_message,
        user_message
    ]
)

script_dir = os.path.dirname(os.path.abspath(__file__))
persistence_dir = os.path.join(script_dir, "..", "chroma")

vectorstore = Chroma(
    embedding_function=ollama_embedder,
    persist_directory=persistence_dir,
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": rag_prompt},
    return_source_documents=True,
    chain_type="stuff"
)


def get_response(query: str) -> str:
    """
    Get the response from the RAG model.
    :param query: The query to ask the model.
    :return: The response from the model.
    """
    if is_toxic(query):
        return blocked_input_response()
    result = rag_chain.invoke({"question": query})
    answer = result["answer"]
    if is_toxic(answer):
        return blocked_input_response()
    return answer


def get_response_with_source(query: str) -> tuple[str, list[str]]:
    """
    Get the response from the RAG model with source documents.
    :param query: The query to ask the model.
    :return: The response from the model with source documents.
    """
    result = rag_chain.invoke({"question": query})
    answer = result["answer"]
    sources = result["source_documents"]
    return answer, [source.page_content for source in sources]
