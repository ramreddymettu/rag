from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser

model = ChatOllama(model="llama3.1")

messages = [
    ("system", "You are an expert product reviewer."),
    ("human", "List the main features of the product {product_name}.")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)

def analyse_pros(features):

    messages = [
        ("system", "You are an expert product reviewer."),
        ("human", "Given the {features}. List the pros the product.")
    ]

    pt = ChatPromptTemplate.from_messages(messages=messages)
    return pt.format_prompt(features= features)


def analyse_cons(features):

    messages = [
        ("system", "You are an expert product reviewer."),
        ("human", "Given the {features}. List the cons the product.")
    ]

    pt = ChatPromptTemplate.from_messages(messages=messages)
    return pt.format_prompt(features= features)

# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

pro_branch_chain = (
    RunnableLambda(lambda x: analyse_pros(x)) | model | StrOutputParser()
)

con_branch_chain = (
    RunnableLambda(lambda x: analyse_cons(x)) | model | StrOutputParser()
)

chain = prompt_template | model | StrOutputParser() | RunnableParallel(branches = {"pros": pro_branch_chain, "cons": con_branch_chain}) | RunnableLambda(lambda x: combine_pros_cons(x['branches']['pros'], x['branches']['cons']))

result = chain.invoke({"product_name": "iPhone 15"})

print(result)