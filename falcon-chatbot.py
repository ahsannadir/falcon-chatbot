# Importing Libraries
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import chainlit as cl

# Loading environment variables from .env file.
load_dotenv()


# Initializing Hugging Face Hub
llm = HuggingFaceHub(
    repo_id = "tiiuae/falcon-7b-instruct",
    model_kwargs = {"temperature": 0.5, "max_new_tokens": 2000}
)

# Prompt Template
template = """
You are a helpful AI assistant. The assistant gives helpful, detailed and polite answers to the user's questions.

{question}
"""

# Chainlit Start Function
@cl.on_chat_start
def main():
    prompt = PromptTemplate(template = template, input_variables = ['question'])
    llm_chain = LLMChain(prompt = prompt, llm = llm)

    cl.user_session.set("llm_chain", llm_chain)

    return llm_chain


# Chainlit Message Fucntion
@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")

    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res["text"]).send()