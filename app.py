from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import torch
import torch.quantization
import torch.nn.utils.prune as prune
from torch.cuda.amp import autocast
from transformers import T5ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

def fix_state_dict(state_dict):
    fixed_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    return fixed_state_dict

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)
    checkpoint = torch.load("./checkpoint.ckpt", map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    return tokenizer, model

tokenizer, model = load_model()

def summarize(text):
    with torch.no_grad(), autocast(enabled=device=='cuda'):
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

st.title("T5 Text Summarization App")

text = st.text_area("Text to summarize", "Paste your text here...", height=300)
if st.button("Summarize"):
    summary = summarize(text)
    st.subheader("Summary")
    st.write(summary)
# streamlit run app.py server.port 40 --server.address 0.0.0.0