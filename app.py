import os
import streamlit as st
import fitz  # PyMuPDF for reading PDFs
from haystack import Pipeline, Document
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
import time

# Ensure the API key is loaded from environment variables
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not anthropic_api_key and not openai_api_key:
    st.error("API keys for Anthropic and OpenAI are not set in environment variables.")
    st.stop()

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
def initialize_pipeline(documents, model_choice="Anthropic"):
    messages = [
        ChatMessage.from_system("You are a prompt expert who answers questions based on the given documents."),
        ChatMessage.from_user(
            "Here are the documents:\n"
            "{% for d in documents %} \n"
            "    {{d.content}} \n"
            "{% endfor %}"
            "\nAnswer: {{query}}"
        ),
    ]

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("fetcher", LinkContentFetcher())
    rag_pipeline.add_component("converter", HTMLToDocument())
    rag_pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["documents"]))
    
    if model_choice == "Anthropic":
        rag_pipeline.add_component(
            "llm",
            AnthropicChatGenerator(
                api_key=Secret.from_env_var("ANTHROPIC_API_KEY"),
                streaming_callback=print_streaming_chunk,
            ),
        )
    elif model_choice == "OpenAI":
        rag_pipeline.add_component(
            "llm",
            OpenAIChatGenerator(
                model="gpt-3.5-turbo",  # Specify your OpenAI model here
                api_key=Secret.from_env_var("OPENAI_API_KEY"),
                streaming_callback=print_streaming_chunk,
            ),
        )
    else:
        st.error("Invalid model choice. Please select either 'Anthropic' or 'OpenAI'.")
        st.stop()

    rag_pipeline.connect("fetcher", "converter")
    rag_pipeline.connect("converter", "prompt_builder")
    rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

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
    st.session_state.messages = {"Anthropic": [], "OpenAI": []}
if 'current_model' not in st.session_state:
    st.session_state.current_model = "Anthropic"

# Model choice selector
model_choice = st.radio("Select the AI model", options=["Anthropic", "OpenAI"])

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
                result = rag_pipeline.run(
                    data={
                        "fetcher": {"urls": ["https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview"]},
                        "prompt_builder": {"template_variables": {"query": prompt, "documents": documents}, "template": messages},
                    }
                )

                response_content = result['llm']['replies'][0].content

                # Ensure proper bullet point formatting
                response_content = response_content.replace("•", "\n- ")

                # Additional formatting for Streamlit display
                response_content = "<div style='margin-left: 20px;'><ul>" + ''.join(
                    [f"<li>{line.strip()}</li>" for line in response_content.split("- ") if line.strip()]
                ) + "</ul></div>"

                st.markdown(response_content, unsafe_allow_html=True)
                st.session_state.messages[st.session_state.current_model].append({"role": "assistant", "content": response_content})

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

else:
    st.warning("Please upload a PDF file to start the conversation.")

st.markdown("---")
st.write("This application utilizes Anthropic Claude's AI model or OpenAI's GPT model to provide ethical reasoning and guidance based on content from uploaded PDF documents.")