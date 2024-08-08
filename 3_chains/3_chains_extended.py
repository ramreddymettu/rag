from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence


model = ChatOllama(model="llama3.1")

messages = [
    ("system", "You are comedian who tells jokes about {topic}."),
    ("human", "Tell me {jc} jokes.")
]


prompt = ChatPromptTemplate.from_messages(messages=messages)
upper_words = RunnableLambda(lambda x: x.upper())
count_runnable = RunnableLambda(lambda x: len(x.split()))

chain = prompt | model | StrOutputParser() | upper_words #| count_runnable

result = chain.invoke({"topic": "Lawyers", "jc": 3})

print(result)
