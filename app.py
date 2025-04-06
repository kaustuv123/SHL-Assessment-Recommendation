import streamlit as st
from engine import setup_engine, get_top_k_recommendations
import pandas as pd
import logging
import sys
import time
import traceback

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SHL-Recommender")

st.set_page_config(page_title="SHL Test Recommender", layout="wide")

logger.info("Starting SHL Test Recommender app")

# Display app title
st.title("🔍 SHL Semantic Test Recommender")
query = st.text_input("Enter your job description or requirement:")

# Initialize the engine
if "engine_ready" not in st.session_state:
    logger.info("Engine not initialized. Starting initialization...")
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    try:
        with st.spinner("Setting up engine..."):
            status_placeholder.text("Loading model and data...")
            start_time = time.time()
            
            # Log Python and dependency versions
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Pandas version: {pd.__version__}")
            
            # Setup engine with progress updates
            progress_bar.progress(10)
            status_placeholder.text("Loading model and data (10%)...")
            
            model, index, data = setup_engine()
            
            progress_bar.progress(90)
            status_placeholder.text("Finalizing setup (90%)...")
            
            st.session_state.update({
                "engine_ready": True,
                "model": model,
                "index": index,
                "data": data
            })
            
            elapsed_time = time.time() - start_time
            logger.info(f"Engine initialized successfully in {elapsed_time:.2f} seconds")
            progress_bar.progress(100)
            status_placeholder.text(f"Setup complete! Loaded {len(data)} assessments.")
            time.sleep(1)  # Give user time to see the completion message
            status_placeholder.empty()
            progress_bar.empty()
    except Exception as e:
        error_msg = f"Error initializing engine: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(f"❌ {error_msg}")
        st.error("Check the logs for more details.")

# Process query if engine is ready
if query and "engine_ready" in st.session_state and st.session_state["engine_ready"]:
    logger.info(f"Processing query: '{query}'")
    try:
        with st.spinner("Searching..."):
            start_time = time.time()
            
            results = get_top_k_recommendations(
                query,
                st.session_state["model"],
                st.session_state["index"],
                st.session_state["data"]
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Query processed in {elapsed_time:.2f} seconds, found {len(results)} results")
            
            if results:
                df = pd.DataFrame(results)[[
                    'assessment_name', 'url', 'remote_testing', 'adaptive_irt', 'duration', 'test_type'
                ]]
                df.columns = [
                    "Assessment Name", "URL", "Remote Testing", "Adaptive/IRT", "Duration", "Test Type"
                ]
                df["Assessment Name"] = df.apply(lambda row: f"[{row['Assessment Name']}]({row['URL']})", axis=1)
                df.drop(columns=["URL"], inplace=True)
                
                st.markdown(f"### 🔗 Top Recommendations (found in {elapsed_time:.2f}s):")
                st.write(df.to_markdown(index=False), unsafe_allow_html=True)
            else:
                st.warning("No matching assessments found for your query.")
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(f"❌ {error_msg}")
        st.error("Check the logs for more details.")

# Add footer with deployment information
st.markdown("---")
st.markdown("#### App Status Information")
if "engine_ready" in st.session_state and st.session_state["engine_ready"]:
    st.success("✅ Engine loaded successfully")
    st.info(f"📊 Loaded {len(st.session_state['data'])} assessments")
else:
    st.error("❌ Engine not loaded")
