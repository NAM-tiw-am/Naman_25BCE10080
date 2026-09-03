from langchain_groq import ChatGroq          
from langchain_core.prompts import ChatPromptTemplate
from retriever import retrieve
import os

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question based ONLY on the following context.
If the context does not contain enough information to answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

def get_llm():
    
    return ChatGroq(                          
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),   
        temperature=0.3,
    )


def format_context(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def generate(query, k=5):
    retrieved_docs = retrieve(query, k=k)
    context = format_context(retrieved_docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format_messages(context=context, question=query)

    # Step 4: Call the LLM
    llm = get_llm()
    response = llm.invoke(formatted_prompt)

    # Step 5: Return the answer text
    return response.content