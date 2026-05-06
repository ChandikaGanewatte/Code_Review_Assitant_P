import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("🧠 Intelligent Code Review Assistant")

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()

labels_map = {
    0: "✅ Clean Code",
    1: "⚠ Inefficient Code",
    2: "🚨 Bug Risk",
    3: "⚠ Code Smell"
}

def predict(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

code = st.text_area("Paste your code here")

if st.button("Analyze"):
    if code.strip() == "":
        st.warning("Please enter code")
    else:
        label = predict(code)
        st.success(labels_map[label])
