
#### conversation bffer memory will takes all the past interactions from humans and it passes through history parameters as the raw text and no processing the data


import inspect
from getpass import getpass

from langchain_community.llms import Ollama
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationKGMemory
)
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import OpenAI
from dotenv import load_dotenv

import os
from langchain.callbacks import tracing_v2_enabled

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize the LLM (Ollama's Llama2)
llm = Ollama(model="llama2")

def token_count(chain, query):
    if tracing_v2_enabled():  # Replace the deprecated get_openai_callback
        result = chain.invoke({"input": query})  # Use invoke instead of run
        print(f"Total Tokens: 0 tokens")  # Change this as needed if token count is available
        return result 


con_buf=ConversationChain(llm=llm,memory=ConversationBufferMemory())
res=con_buf.invoke({"input":"Hi how are you ai!"})
res1=token_count(con_buf,"My interest is to explore the potential of Large Language Models that can process NLP and generate text.")
res2=token_count(con_buf,"I just want to analyze the different possibilities. What can you think of?")
res3=token_count(con_buf,"What is my goal again?")




