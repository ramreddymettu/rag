from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

#Load Env Files
load_dotenv()

# OpenAI chat model
model = ChatOllama(model="llama3.1")

#Invoke the model with a message
result = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)