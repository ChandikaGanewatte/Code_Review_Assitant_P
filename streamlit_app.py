import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Intelligent Code Review Assistant",
    page_icon="🧠",
    layout="centered"
)

# ============================================
# TITLE
# ============================================
st.title("🧠 Intelligent Code Review Assistant")
st.markdown("Analyze Python code for bugs, inefficiencies, and code smells using AI + rule-based analysis.")

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base",
        num_labels=4
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/codebert-base"
    )

    return model, tokenizer


model, tokenizer = load_model()

# ============================================
# LABELS
# ============================================
labels_map = {
    0: "✅ Clean Code",
    1: "⚠ Inefficient Code",
    2: "🚨 Bug Risk",
    3: "⚠ Code Smell"
}

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict(code):

    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return prediction


# ============================================
# RULE-BASED ANALYSIS
# ============================================
def analyze_code_rules(code):

    issues = []

    # Infinite loop
    if "while True" in code:
        issues.append((
            "🚨 Bug Risk",
            "Possible infinite loop detected. Add a break condition."
        ))

    # Nested loops
    if code.count("for") > 1:
        issues.append((
            "⚠ Inefficient",
            "Nested loops detected. Consider optimizing complexity."
        ))

    # None comparison
    if "== None" in code or "!= None" in code:
        issues.append((
            "⚠ Code Smell",
            "Use 'is None' instead of '== None'."
        ))

    # Large function
    if len(code) > 800:
        issues.append((
            "⚠ Maintainability",
            "Function too long. Break into smaller functions."
        ))

    # Too many prints
    if code.count("print") > 5:
        issues.append((
            "⚠ Debug Code",
            "Too many print statements. Use logging instead."
        ))

    # Try without exception handling
    if "try:" in code and "except" not in code:
        issues.append((
            "⚠ Error Handling",
            "Try block found without exception handling."
        ))

    # Hardcoded password
    if "password =" in code.lower():
        issues.append((
            "🚨 Security Risk",
            "Hardcoded password detected."
        ))

    return issues


# ============================================
# SUGGESTIONS
# ============================================
def generate_suggestions(code, model_label):

    suggestions = []

    # Model-based suggestions
    if model_label == 1:
        suggestions.append(
            "Optimize loops and reduce algorithm complexity."
        )

    elif model_label == 2:
        suggestions.append(
            "Check for unsafe conditions or infinite loops."
        )

    elif model_label == 3:
        suggestions.append(
            "Improve coding style and follow PEP8 practices."
        )

    else:
        suggestions.append(
            "Code structure looks good."
        )

    # Rule-based suggestions
    issues = analyze_code_rules(code)

    for issue in issues:
        suggestions.append(issue[1])

    return suggestions


# ============================================
# SAMPLE CODE BUTTONS
# ============================================
st.subheader("📌 Sample Test Codes")

sample1 = """
def sum_list(arr):
    total = 0
    for num in arr:
        total += num
    return total
"""

sample2 = """
while True:
    for i in range(10):
        for j in range(10):
            print(i, j)
"""

sample3 = """
def check(value):
    if value == None:
        return "Empty"
    return "Not Empty"
"""

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Clean Code"):
        st.session_state.code = sample1

with col2:
    if st.button("Bug Risk"):
        st.session_state.code = sample2

with col3:
    if st.button("Code Smell"):
        st.session_state.code = sample3


# ============================================
# CODE INPUT
# ============================================
default_code = st.session_state.code if "code" in st.session_state else ""

code = st.text_area(
    "Paste your Python code here:",
    value=default_code,
    height=300
)

# ============================================
# ANALYZE BUTTON
# ============================================
if st.button("🔍 Analyze Code"):

    if code.strip() == "":
        st.warning("Please enter Python code.")

    else:

        # Prediction
        label = predict(code)

        # ==========================
        # RESULT
        # ==========================
        st.subheader("🔍 AI Prediction")
        st.success(labels_map[label])

        # ==========================
        # RULE ANALYSIS
        # ==========================
        st.subheader("🧠 Detailed Rule-Based Analysis")

        issues = analyze_code_rules(code)

        if issues:
            for issue_type, message in issues:
                st.warning(f"{issue_type}: {message}")

        else:
            st.info("No major rule-based issues detected.")

        # ==========================
        # SUGGESTIONS
        # ==========================
        st.subheader("💡 Suggestions")

        suggestions = generate_suggestions(code, label)

        for s in suggestions:
            st.write(f"✔ {s}")

        # ==========================
        # METRICS DISPLAY
        # ==========================
        st.subheader("📊 Analysis Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Lines of Code", len(code.split("\n")))

        with col2:
            st.metric("Characters", len(code))

        with col3:
            st.metric("Detected Issues", len(issues))


# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "Built using CodeBERT, Transformers, PyTorch, and Streamlit."
)
