from __future__ import annotations

from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.answer_generator import answer_question
from app.ui.components import render_citations


st.set_page_config(page_title="FFXIV RAG Assistant", layout="wide")
st.title("FFXIV Multimodal RAG Assistant")

question = st.text_input("Question", "How should I prepare for raid mitigation?")
if st.button("Ask") and question:
    result = answer_question(question)
    st.write(result["answer"])
    render_citations(result["citations"])
