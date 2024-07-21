import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# Model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)

# Function to preprocess the file
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# Function to perform summarization
def llm_summarization(text):
    pipe_sum = pipeline('summarization',
                        model=base_model,
                        tokenizer=tokenizer,
                        max_length=500,
                        min_length=50)
    result = pipe_sum(text)
    return result[0]['summary_text']

# Function to perform question answering
def llm_qna(context, question):
    pipe_qna = pipeline('question-answering',
                        model=base_model,
                        tokenizer=tokenizer)
    answer = pipe_qna({'context': context, 'question': question})
    return answer['answer']

# Function to display the PDF
@st.cache
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to display the answer
def display_answer(answer):
    st.info("Question Answering Complete")
    st.success(f"Answer: {answer}")

# Streamlit code
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization and Q&A")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        filepath = "data/" + uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        st.info("Uploaded File")
        displayPDF(filepath)

        summary = llm_summarization(file_preprocessing(filepath))
        st.info("Summarization Complete")
        st.success(summary)

        user_question = st.text_input("Ask a question about the PDF content:")
        if user_question:
            context = file_preprocessing(filepath)
            answer = llm_qna(context, user_question)
            display_answer(answer)

if __name__ == "__main__":
    main()
