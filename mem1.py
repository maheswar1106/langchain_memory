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
from langchain.callbacks import tracing_v2_enabled
from langchain import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize the LLM (Ollama's Llama2)
llm = Ollama(model="llama2")

# Function to calculate token count using the invoke method
def token_count(chain, query):
    if tracing_v2_enabled():  # Replace deprecated get_openai_callback
        result = chain.invoke({"input": query})  # Use invoke instead of run
        print(f"Total Tokens: 0 tokens")  # Adjust if token count is available
        return result

# Initialize the conversation chain with ConversationSummaryMemory
con_sum = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm))

# Start the conversation and count tokens
res = token_count(con_sum, "Good morning AI!")
res1 = token_count(con_sum, "My interest is to explore the potential of Large Language Models that can process NLP and generate text.")
res2 = token_count(con_sum, "I just want to analyze the different possibilities. What can you think of?")
res3 = token_count(con_sum, "What is my aim again?")

# Print the memory buffer to see the conversation summary
print(con_sum.memory.buffer)
