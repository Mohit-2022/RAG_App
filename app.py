import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


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
    prompt_template = """Use the context below to answer
the question. If the answer requires any calculation
such as addition or subtraction of numbers mentioned
in the context, calculate and give the result.
If answer is not in context say:
"This information is not in the policy document.
 Please contact HR."

    Context: {context}

    Question: {question}

    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Build chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

# Load the chain
chain = load_chain()
st.success("✅ Leave Policy loaded! Ask your question below.")

# ─────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ─────────────────────────────────────────
# USER INPUT
# ─────────────────────────────────────────
if question := st.chat_input("Ask about leave policy..."):

    # Show user question
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching policy..."):
            result = chain({"query": question})
            answer = result["result"]
            st.write(answer)

            # Show source pages
            with st.expander("📄 View source sections"):
                for i, doc in enumerate(
                    result["source_documents"]
                ):
                    st.markdown(
                        f"**Source {i+1} — "
                        f"Page {doc.metadata['page']+1}:**"
                    )
                    st.info(doc.page_content[:300])

    # Save answer
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
