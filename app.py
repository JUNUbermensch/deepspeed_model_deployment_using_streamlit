from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import torch

model_name_or_path = "eenzeenee/t5-base-korean-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def summarize(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

st.title("Text Summarization App")

text = st.text_area("Text to summarize", "Paste your text here...", height=300)
if st.button("Summarize"):
    summary = summarize(text)
    st.subheader("Summary")
    st.write(summary)
# streamlit run app.py --server.address 0.0.0.0