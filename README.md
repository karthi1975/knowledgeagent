
# README for Ethical Agent with PDF Application

## Overview

The "Ethical Agent with PDF" application is a Streamlit-based web app designed to assist users in extracting and interacting with content from PDF documents. The application leverages either Anthropic's Claude AI model or OpenAI's GPT model to provide ethical reasoning and guidance based on the content extracted from the uploaded PDF documents.

## Features

- **PDF Document Processing**: Upload a PDF file, and the application reads and preprocesses its content for AI-driven analysis.
- **AI Model Integration**: Choose between Anthropic's Claude or OpenAI's GPT-3.5-turbo for generating responses.
- **Interactive Chat Interface**: Engage in a conversation with the AI model, asking questions related to the content of the uploaded PDF.
- **Streamlit Interface**: A user-friendly web interface built using Streamlit, which includes options for file uploading, model selection, and displaying chat responses.

## Prerequisites

- **Python 3.7+**: Ensure you have Python installed on your machine.
- **API Keys**: Obtain API keys for Anthropic and OpenAI models and set them as environment variables:
  - `ANTHROPIC_API_KEY`
  - `OPENAI_API_KEY`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/ethical-agent-pdf.git
   cd ethical-agent-pdf
   ```

2. **Install Required Packages**:
   Install all necessary Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

   The required packages include:
   - `streamlit`
   - `PyMuPDF` (fitz)
   - `haystack`
   - `haystack-integrations`
   - `time`

3. **Set Up Environment Variables**:
   Ensure your API keys are set in the environment. For example, on Linux or macOS, you can add the following to your `.bashrc` or `.zshrc`:
   ```bash
   export ANTHROPIC_API_KEY="your_anthropic_api_key"
   export OPENAI_API_KEY="your_openai_api_key"
   ```

   On Windows, you can set environment variables using the Command Prompt:
   ```cmd
   setx ANTHROPIC_API_KEY "your_anthropic_api_key"
   setx OPENAI_API_KEY "your_openai_api_key"
   ```

## Usage

1. **Run the Application**:
   Start the Streamlit app by running the following command in your terminal:
   ```bash
   streamlit run app.py
   ```

2. **Upload a PDF Document**:
   - Use the file uploader to upload a PDF file.
   - Alternatively, use the default PDF provided with the application.

3. **Select AI Model**:
   - Choose between "Anthropic" or "OpenAI" as the AI model for generating responses.

4. **Interact with the AI**:
   - Enter your questions related to the uploaded PDF content in the chat input field.
   - The AI will provide responses based on the content extracted from the PDF and its model's capabilities.

## Application Components

- **PDF Reader**: Utilizes `PyMuPDF` to read and extract text from PDF documents.
- **Haystack Pipeline**: Configures a Retrieval-Augmented Generation (RAG) pipeline to fetch content, convert it into a readable format, build prompts, and generate responses using the selected AI model.
- **Chat Interface**: A dynamic chat interface for users to interact with the AI model and receive formatted responses.

## Error Handling

- The application checks for the presence of API keys and displays an error message if they are not set.
- If a user uploads a non-PDF file or an empty PDF, the application will provide appropriate warnings.

## Future Enhancements

- Add support for other document formats like DOCX or HTML.
- Implement advanced formatting options for better visualization of responses.
- Extend the AI model options to include more generative models.

## Contributing

We welcome contributions! Please fork the repository and submit a pull request for any enhancements or bug fixes.



## Contact

For further questions or support, please contact [karthi.jeyabalan@gmail.com].
