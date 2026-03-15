import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



st.set_page_config(
    page_title="Leave Policy Assistant",
    page_icon="📋",
    layout="centered"
)

st.title("Leave Policy Assistant")
st.markdown("Ask me anything about the **company leave policy!**")
st.divider()


@st.cache_resource
def load_chain():

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load saved vectorstore
    vectorstore = FAISS.load_local(
        "leave_policy_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Load Gemini
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        google_api_key=st.secrets["OPENAI_API_KEY"],
        temperature=0,
        convert_system_message_to_human=True
    )

    # Custom prompt
    
    prompt = PromptTemplate.from_template("""
You are a helpful HR assistant.
Use the context below to answer the question.
If calculation needed, calculate and give result.
Answer in 2-3 sentences clearly.
If not found say: "Not in policy. Please contact HR."

Context: {context}
Question: {question}
Answer:""")

    # Build chain
  
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever
chain, retriever = load_chain()

st.success("✅ Leave Policy loaded! Ask your question below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask about leave policy..."):
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching policy..."):
            try:
                answer = chain.invoke(question)
                st.write(answer)
                with st.expander("📄 View source sections"):
                    docs = retriever.invoke(question)
                    for i, doc in enumerate(docs):
                        st.markdown(
                            f"**Source {i+1} — "
                            f"Page {doc.metadata['page']+1}:**"
                        )
                        st.info(doc.page_content[:300])
            except Exception as e:
                st.warning("⏳ Please try again!")
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
