from __future__ import annotations

import streamlit as st


def render_citations(citations: list[dict]) -> None:
    st.subheader("Citations")
    if not citations:
        st.info("No citations.")
        return
    st.dataframe(citations, use_container_width=True)
