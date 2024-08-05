from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.1")

chat_history = []

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

while True:
    query = input("You: ")

    if query.lower() == "/exit":
        break

    human_message = HumanMessage(content=query)
    chat_history.append(human_message)
    response = model.invoke(chat_history)
    print("AI: ", response.content)

    chat_history.append(AIMessage(content=response.content))