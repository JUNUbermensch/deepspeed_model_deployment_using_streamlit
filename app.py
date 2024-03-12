from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import torch
import torch.quantization
import torch.nn.utils.prune as prune
from torch.cuda.amp import autocast
from transformers import T5ForConditionalGeneration

device = "cpu"
model_path = './t5_mss_small_torch'

def fix_state_dict(state_dict):
    fixed_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    return fixed_state_dict

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    checkpoint_path = "./checkpoint.ckpt"
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        fixed_state_dict = {key.replace('module.', ''): value for key, value in checkpoint.items()}
        model.load_state_dict(fixed_state_dict, strict=False)
    
    model.to(device)
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    parameters_to_prune = (
        (model.encoder.block[0].layer[1].layer_norm, 'weight'),
    )
    
    for module, param_name in parameters_to_prune:
        prune.l1_unstructured(module, name=param_name, amount=0.2)
    
    return tokenizer, model

tokenizer, model = load_model()

def summarize(text):
    with torch.no_grad(), autocast(enabled=device == 'cuda'):
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

st.title("Text Summarization with Pruned T5")
text = st.text_area("Enter text:", "Paste your text here...", height=300)
if st.button("Summarize"):
    summary = summarize(text)
    st.subheader("Summary")
    st.write(summary)
    
# streamlit run app.py server.port 40 --server.address 0.0.0.0