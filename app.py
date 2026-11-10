
import streamlit as st

from src.rag_pipeline import RAGPipeline


# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------
# INITIALIZE RAG PIPELINE
# ---------------------------------

@st.cache_resource
def load_rag_pipeline():

    return RAGPipeline()


# ---------------------------------
# APPLICATION TITLE
# ---------------------------------

st.title("🤖 Enterprise RAG Assistant")

st.write(
    "Ask questions about company policies, "
    "travel, leave, and IT support."
)

st.divider()


# ---------------------------------
# SIDEBAR
# ---------------------------------

with st.sidebar:

    st.header("About")

    st.write(
        "This application uses Retrieval-Augmented "
        "Generation (RAG) and Gemini AI to answer "
        "questions using company documents."
    )

    st.divider()

    st.subheader("Features")

    st.write("📄 Company document search")
    st.write("🔍 Semantic retrieval")
    st.write("🤖 Gemini AI answers")
    st.write("📚 Supporting sources")

    if st.button("Clear Chat"):

        st.session_state.messages = []
        st.rerun()


# ---------------------------------
# LOAD RAG PIPELINE
# ---------------------------------

try:

    rag = load_rag_pipeline()

except Exception as error:

    st.error(f"Unable to initialize RAG: {error}")
    st.stop()


# ---------------------------------
# INITIALIZE CHAT HISTORY
# ---------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------
# DISPLAY PREVIOUS MESSAGES
# ---------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander("View supporting sources"):

                    for source in sources:

                        st.write(
                            f"📄 {source['source']}"
                        )

                        st.caption(
                            f"Department: {source['department']}"
                        )


# ---------------------------------
# USER CHAT INPUT
# ---------------------------------

question = st.chat_input(
    "Ask a question about company policies..."
)


if question:

    # Display user question

    st.session_state.messages.append({

        "role": "user",
        "content": question

    })

    with st.chat_message("user"):

        st.markdown(question)


    # Generate RAG answer

    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            try:

                result = rag.ask(
                    question=question,
                    top_k=3,
                    min_score=0.40
                )

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                # Display supporting documents

                if sources:

                    with st.expander("View supporting sources"):

                        for source in sources:

                            st.write(
                                f"📄 {source['source']}"
                            )

                            st.caption(
                                f"Department: {source['department']}"
                            )

                else:

                    st.info(
                        "No supporting company documents were retrieved."
                    )

                # Save assistant response

                st.session_state.messages.append({

                    "role": "assistant",
                    "content": answer,
                    "sources": sources

                })

            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )
