import streamlit as st, torch, gc, torch.quantization, torch.nn.utils.prune as prune
from transformers import AutoTokenizer, T5ForConditionalGeneration

device, model_path = "cpu", 'wisenut-nlp-team/t5-fid-new'

def load_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    parameters_to_prune = ((model.encoder.block[0].layer[1].layer_norm, 'weight'),)
    for module, param_name in parameters_to_prune: prune.l1_unstructured(module, name=param_name, amount=0.2)
    return tokenizer, model

def summarize(text):
    with torch.no_grad():
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def unload_model():
    global model, tokenizer
    model = None
    tokenizer = None
    torch.cuda.empty_cache() # If using GPU
    gc.collect()

st.title("Text Summarization with Pruned T5")
text = st.text_area("Enter text:", "Paste your text here...", height=300)
if st.button("Summarize"):
    tokenizer, model = load_model()
    summary = summarize(text)
    st.subheader("Summary")
    st.write(summary)
    unload_model()
    
# streamlit run app.py server.port 8501 --server.address 0.0.0.0