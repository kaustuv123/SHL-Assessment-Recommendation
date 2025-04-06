import streamlit as st
from engine import setup_engine, get_top_k_recommendations, enrich_query_with_gemini
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="SHL Test Recommender", layout="wide")

st.title("🔍 SHL Semantic Test Recommender")
query = st.text_input("Enter your job description or requirement:")

if "engine_ready" not in st.session_state:
    with st.spinner("Setting up engine..."):
        model, index, data, gemini_api_key = setup_engine()
        st.session_state.update({
            "engine_ready": True,
            "model": model,
            "index": index,
            "data": data,
            "gemini_api_key": gemini_api_key
        })

if query and st.session_state["engine_ready"]:
    with st.spinner("Searching..."):
        # Enrich the query using Gemini if API key is available
        if st.session_state["gemini_api_key"]:
            with st.spinner("Enhancing your query..."):
                enriched_query = enrich_query_with_gemini(query, st.session_state["gemini_api_key"])
                st.info(f"🔍 Enhanced query: {enriched_query}")
        else:
            enriched_query = query
            st.warning("⚠️ Query enrichment disabled - GEMINI_API_KEY not found")
        
        results = get_top_k_recommendations(
            enriched_query,
            st.session_state["model"],
            st.session_state["index"],
            st.session_state["data"]
        )
        df = pd.DataFrame(results)[[
            'assessment_name', 'url', 'remote_testing', 'adaptive_irt', 'duration', 'test_type'
        ]]
        
        # Rename columns for display
        df.columns = [
            "Assessment Name", "URL", "Remote Testing", "Adaptive/IRT", "Duration", "Test Type"
        ]
        
        st.markdown("### 🔗 Top Recommendations:")
        
        # Configure the DataFrame display with clickable links
        st.dataframe(
            df,
            column_config={
                "Assessment Name": st.column_config.LinkColumn(
                    "Assessment Name",
                    help="Click to view the assessment",
                    validate="^https?://.*",
                ),
                "URL": st.column_config.Column(
                    "URL",
                    help="Direct link to the assessment",
                    disabled=True,
                ),
                "Remote Testing": st.column_config.TextColumn(
                    "Remote Testing",
                    help="Whether remote testing is supported",
                ),
                "Adaptive/IRT": st.column_config.TextColumn(
                    "Adaptive/IRT",
                    help="Whether adaptive/IRT is supported",
                ),
                "Duration": st.column_config.TextColumn(
                    "Duration",
                    help="Assessment duration",
                ),
                "Test Type": st.column_config.TextColumn(
                    "Test Type",
                    help="Type of assessment",
                ),
            },
            hide_index=True,
        )
