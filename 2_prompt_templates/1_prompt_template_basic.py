from langchain_core.prompts import ChatPromptTemplate


# Template
template = "Tell me a joke about {topic}."

prompt_template = ChatPromptTemplate.from_template(template=template)

prompt = prompt_template.invoke({"topic": "cats"})

print("-------- Input Prompt-------")
#print(prompt)

## Prompt with System and Human Messages with ChatPrompt

messages = [
    ("system", "You are comedian who tells jokes about {topic}."),
    ("human", "Tell me {jc} jokes.")
]

template = ChatPromptTemplate.from_messages(messages=messages)
prompt = template.invoke({"topic": "Lawyers", "jc": 3})

print(prompt)

