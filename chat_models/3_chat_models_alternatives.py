from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

## Different models
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

## Messages
messages = [
    SystemMessage(content="Solve the following math problems. Only respond with the answer."),
    HumanMessage(content="What is 81 divided by 9?.")
]


# Create ChatOllama Models
model = ChatOllama(model="llama3.1")

result = model.invoke(messages)
print("Resonse from AI: ", result.content)


# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")


# ---- Anthropic Chat Model Example ----

# Create a Anthropic model
# Anthropic models: https://docs.anthropic.com/en/docs/models-overview
model = ChatAnthropic(model="claude-3-opus-20240229")

result = model.invoke(messages)
print(f"Answer from Anthropic: {result.content}")


# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

result = model.invoke(messages)
print(f"Answer from Google: {result.content}")
