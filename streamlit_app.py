import streamlit as st
from engine import setup_engine, get_top_k_recommendations
import pandas as pd

st.set_page_config(page_title="SHL Test Recommender", layout="wide")

st.title("🔍 SHL Semantic Test Recommender")
query = st.text_input("Enter your job description or requirement:")

if "engine_ready" not in st.session_state:
    with st.spinner("Setting up engine..."):
        model, index, data = setup_engine()
        st.session_state.update({
            "engine_ready": True,
            "model": model,
            "index": index,
            "data": data
        })

if query and st.session_state["engine_ready"]:
    with st.spinner("Searching..."):
        results = get_top_k_recommendations(
            query,
            st.session_state["model"],
            st.session_state["index"],
            st.session_state["data"]
        )
        df = pd.DataFrame(results)[[
            'assessment_name', 'url', 'remote_testing', 'adaptive_irt', 'duration', 'test_type'
        ]]
        df.columns = [
            "Assessment Name", "URL", "Remote Testing", "Adaptive/IRT", "Duration", "Test Type"
        ]
        df["Assessment Name"] = df.apply(lambda row: f"[{row['Assessment Name']}]({row['URL']})", axis=1)
        df.drop(columns=["URL"], inplace=True)
        st.markdown("### 🔗 Top Recommendations:")
        st.write(df.to_markdown(index=False), unsafe_allow_html=True)
