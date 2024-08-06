from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

model = ChatOllama(model="llama3.1")

# Template 1
template = "Tell me a 3 jokes about {topic}."
prompt_template = ChatPromptTemplate.from_template(template=template)
prompt = prompt_template.invoke({"topic": "cats"})

# response = model.invoke(prompt)
# print(response.content)

messages = [
    ("system", "You are comedian who tells jokes about {topic}."),
    ("human", "Tell me {jc} jokes.")
]

template = ChatPromptTemplate.from_messages(messages=messages)
prompt = template.invoke({"topic": "Lawyers", "jc": 3})

response = model.invoke(prompt)
print(response.content)