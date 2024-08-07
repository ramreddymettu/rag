from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence


model = ChatOllama(model="llama3.1")

messages = [
    ("system", "You are comedian who tells jokes about {topic}."),
    ("human", "Tell me {jc} jokes.")
]

prompt = ChatPromptTemplate.from_messages(messages=messages)
prompt_runnable = RunnableLambda(lambda x: prompt.format_prompt(**x))
model_runnable = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=prompt_runnable, middle=[model_runnable], last=parse_output)

result = chain.invoke({'topic': 'Lawyers', 'jc': 3})

print(result)