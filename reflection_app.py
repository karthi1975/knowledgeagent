import os
import streamlit as st
import fitz  # PyMuPDF for reading PDFs
from haystack import Pipeline, Document
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.utils import Secret
from transformers import pipeline

# Function to read and preprocess documents from a PDF file
def read_documents_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    documents = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():  # Ensure that the page has text
            documents.append(Document(content=text))
    return documents

# Initialize the components and pipeline for RAG
def initialize_pipeline(documents, model_choice="Reflection-Llama"):
    messages = [
        {"role": "system", "content": "You are a prompt expert who answers questions based on the given documents."},
        {"role": "user", "content": "Here are the documents:\n" + "\n".join([d.content for d in documents]) + "\nAnswer: {{query}}"}
    ]

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("fetcher", LinkContentFetcher())
    rag_pipeline.add_component("converter", HTMLToDocument())
    rag_pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["documents"]))

    if model_choice == "Reflection-Llama":
        reflection_pipeline = pipeline("text-generation", model="mattshumer/Reflection-Llama-3.1-70B")
        return reflection_pipeline, messages

    else:
        st.error("Invalid model choice. Please select 'Reflection-Llama'.")
        st.stop()

    return rag_pipeline, messages

# Streamlit Chat Interface
st.title("Ethical Agent with PDF")

# Provide download link for the default PDF
default_pdf_path = os.path.join(os.getcwd(), "principals_ethic_ai.pdf")
if os.path.exists(default_pdf_path):
    with open(default_pdf_path, "rb") as f:
        st.download_button(
            label="Download default PDF (principals_ethic_ai.pdf)",
            data=f,
            file_name="principals_ethic_ai.pdf",
            mime="application/pdf"
        )
else:
    st.error("Default PDF not found. Please upload a PDF file.")

# Initialize session state for messages if not already done
if 'messages' not in st.session_state:
    st.session_state.messages = {"Reflection-Llama": []}
if 'current_model' not in st.session_state:
    st.session_state.current_model = "Reflection-Llama"

# Model choice selector
model_choice = st.radio("Select the AI model", options=["Reflection-Llama"])

# Update the current model in session state if it changes
if model_choice != st.session_state.current_model:
    st.session_state.current_model = model_choice

# Automatically load the default PDF if available
if os.path.exists(default_pdf_path):
    with open(default_pdf_path, "rb") as default_pdf:
        documents = read_documents_from_pdf(default_pdf)
    st.success("Default PDF loaded and processed successfully!")
else:
    documents = []

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a different PDF file", type=["pdf"])

if uploaded_file:
    documents = read_documents_from_pdf(uploaded_file)
    st.success("PDF uploaded and processed successfully!")

# Initialize pipeline with the extracted documents and selected model
if documents:
    rag_pipeline, messages = initialize_pipeline(documents, model_choice=st.session_state.current_model)

    # Chat interface
    st.write("## Chat")
    for message in st.session_state.messages[st.session_state.current_model]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages[st.session_state.current_model].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Use the Reflection-Llama model to generate a response
                result = rag_pipeline(messages=[{"role": "user", "content": prompt}])
                response_content = result[0]['generated_text']

                # Display the response
                st.markdown(response_content)
                st.session_state.messages[st.session_state.current_model].append({"role": "assistant", "content": response_content})

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

else:
    st.warning("Please upload a PDF file to start the conversation.")

st.markdown("---")
st.write("This application utilizes the Reflection-Llama model to provide ethical reasoning and guidance based on content from uploaded PDF documents.")