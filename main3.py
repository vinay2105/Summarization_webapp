from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from fpdf import FPDF
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os


load_dotenv("D:\summarization\key.env")  
HUGGINGFACEHUB_API_TOKEN = st.secrets["huggingface"]["api_token"]


def summarization(text):
    llm = HuggingFaceHub(repo_id="utrobinmv/t5_summary_en_ru_zh_base_2048", model_kwargs={"temperature":0,"max_length":64} )

    prompt = PromptTemplate(input_variables=['text'], template='give the summary of given text {text}')
    
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(text)
    return summary
def clean_text(text):
    
    replacements = {
        "\u2018": "'",  
        "\u2019": "'",  
        "\u201C": '"',  
        "\u201D": '"', 
        "\u2013": "-",
        "\u2014": "-", 
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return text

def create_pdf(summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", size=25)
    pdf.cell(0, 20, "SUMMARY", ln=True, align="C")


    pdf.set_font("Arial", size=12)
    cleaned_summary = clean_text(summary)
    pdf.multi_cell(0, 10, cleaned_summary)

    pdf_data = pdf.output(dest="S").encode("latin1")  
    return pdf_data



import streamlit as st
st.title("Text Summarization App")
st.write("Enter text below, and I’ll summarize it for you.")

text = st.text_area("Enter the text to be summarized", height=200)

if st.button("Summarize"):
    if text:
        summary = summarization(text)
        if summary:
            st.subheader("Summary:")
            st.write(summary)
            
            st.write("Download options:")
            txt_button = st.download_button("Download as TXT", data=summary, file_name="summary.txt", mime="text/plain")
            pdf_button = st.download_button("Download as PDF", data=create_pdf(summary), file_name="summary.pdf", mime="application/pdf")