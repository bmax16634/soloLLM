import streamlit as st
from pathlib import Path
from inference.soloGPT_v1_generate import generate
import base64

# ─── two levels up from this file ───
BASE = Path(__file__).resolve().parent.parent
logo_path = BASE / "assets" / "soloLLM2.png"

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

st.set_page_config(page_title="SoloLLM Generator", layout="centered")

# === HEADER with vertical alignment ===
logo_width = 150
logo_base64 = get_base64_of_bin_file(logo_path)

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" width="{logo_width}" style="margin-right: 1rem;" />
        <h2 style="margin: 0;">SoloLLM Generator</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# === INPUT ===
prompt = st.text_area("Enter your prompt:", value="The future of AI is...", height=150)

with st.expander("Advanced settings", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    with col2:
        top_k = st.slider("Top-k Sampling", 1, 100, 40)
    with col3:
        max_tokens = st.slider("Max New Tokens", 10, 500, 100, 10)

# === GENERATE BUTTON ===
if st.button("Generate"):
    with st.spinner("Generating response..."):
        output = generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    st.markdown("### Output")
    st.write(output)
