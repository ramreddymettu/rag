from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


model = ChatOllama(model="llama3.1")

messages = [
    ("system", "You are comedian who tells jokes about {topic}."),
    ("human", "Tell me {jc} jokes.")
]

prompt = ChatPromptTemplate.from_messages(messages=messages)

chain = prompt | model | StrOutputParser()

result = chain.invoke({'topic': 'Lawyers', 'jc': 3})

print(result)