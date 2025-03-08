from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral medium blog writer grading a medium blog. Generate critique and recommendations for the user's blog."
            "Always provide detailed recommendations, including requests for length, virality, style, content, etc.",
        ),
        # in order to fetch history messages
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medium blog writer assistant tasked with writing excellent medium blog posts."
            " Generate the best medium blog post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini")
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm