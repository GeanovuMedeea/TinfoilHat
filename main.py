import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from model import get_response


def response(message, history):
    answer = get_response(message)
    return answer


gr.ChatInterface(
    fn=response,
    type="messages",
    title="TinFoil Hat 📶🛜📡",
    description="Talk to the world's most demented AI!",
    theme="default",
).launch(share=True)
