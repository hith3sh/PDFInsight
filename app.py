import streamlit as st 
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import base64           # to read the pdf
from PIL import Image

# load the model & tokenizer
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

#file loader and preprocessing
def file_preprocessing(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts, len(final_texts)

# LLM pipeline- using summarization pipleine
def llm_pipeline(filepath):
    input_text, input_length = file_preprocessing(filepath)
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = input_length//8, 
        min_length = 25)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

@st.cache_data      #to improve performance by caching
def displayPDF(file):
    # Opening file from file path as read binary
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF file in the web browser
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

#streamlit code 
st.set_page_config(page_title='pdf insight',layout="wide",page_icon="📃",initial_sidebar_state="expanded")
def main():
    st.title("PDF Insight")
    image = Image.open('pdf_logo.jpg')
    st.image(image,width=200)

    uploaded_file = st.file_uploader("Upload the PDF", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns([0.4,0.6])
            filepath = "uploaded_pdfs/"+uploaded_file.name

            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            
            with col1:
                st.info("Uploaded PDF")
                pdf_view = displayPDF(filepath)

            with col2:
                summary = llm_pipeline(filepath)
                st.info("Summarization")
                st.success(summary)


#initializing the app
if __name__ == "__main__":
    main()