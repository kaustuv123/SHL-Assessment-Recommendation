from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
from typing import List, Dict, Any
import os
import logging
from dotenv import load_dotenv
import requests
import time
from huggingface_hub.utils import HfHubHTTPError
from pathlib import Path
import shutil


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and validate the scraped data."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found")
        
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or not data:
            raise ValueError("Invalid data format: expected non-empty list")
        
        # Process all assessments regardless of their type
        processed_data = []
        for item in data:
            # Create a standardized assessment entry
            assessment = {
                'assessment_name': item.get('Title', ''),
                'assessment_type': item.get('Product Type', ''),
                'remote_testing': item.get('Remote Testing', ''),
                'adaptive_irt': item.get('Adaptive/IRT', ''),
                'test_type': item.get('Test Type', ''),
                'description': item.get('Description', ''),
                'job_levels': item.get('Job Levels', ''),
                'languages': item.get('Languages', ''),
                'duration': item.get('Assessment Length', ''),
                'url': item.get('Detail Page', '')
            }
            processed_data.append(assessment)
            
        return processed_data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_embeddings(data: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2") -> tuple[np.ndarray, SentenceTransformer]:
    """Create embeddings from assessment descriptions with additional context."""
    try:
        # Set cache directory to ensure models are saved locally
        import os
        from pathlib import Path
        import time
        from huggingface_hub.utils import HfHubHTTPError
        
        # Create cache directory if it doesn't exist
        cache_dir = Path(os.path.join(os.getcwd(), "model_cache"))
        cache_dir.mkdir(exist_ok=True)
        
        # Try to load model with retries
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading model (attempt {attempt+1}/{max_retries})...")
                model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
                break
            except HfHubHTTPError as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    # If we've exhausted retries or it's not a rate limit error
                    if "429" in str(e):
                        logger.error("Hugging Face rate limit exceeded. Manual workaround required:")
                        logger.error("1. Download the model manually from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
                        logger.error(f"2. Save it to {cache_dir}/sentence-transformers_all-MiniLM-L6-v2")
                    raise
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
        
        # Create rich text descriptions that include multiple fields
        rich_descriptions = []
        for item in data:
            rich_text = f"""
            Assessment Type: {item['assessment_type']}
            Test Type: {item['test_type']}
            Job Levels: {item['job_levels']}
            Description: {item['description']}
            """
            rich_descriptions.append(rich_text.strip())
        
        embeddings = model.encode(rich_descriptions, show_progress_bar=True)
        return embeddings, model
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create and populate FAISS index."""
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        raise

def get_top_k_recommendations(
    query_text: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    metadata: List[Dict[str, Any]],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Get top-k recommendations based on query text."""
    try:
        # Enhance the query with context about what we're looking for
        enhanced_query = f"Find assessments matching: {query_text}"
        query_embedding = model.encode([enhanced_query])
        distances, indices = index.search(np.array(query_embedding), top_k)
        results = [metadata[i] for i in indices[0]]
        return results
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise

import requests

def enrich_query_with_gemini(prompt: str, api_key: str) -> str:
    """Uses Gemini Pro API to expand the query intelligently."""
    try:
        endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": f"""You are an expert SHL assessment matcher. Expand this job query with relevant keywords and skills. Focus on job level, languages, assessment types etc.

User Query: {prompt}
Return only the enriched query.
"""}]
            }]
        }
        response = requests.post(f"{endpoint}?key={api_key}", headers=headers, json=payload)
        response.raise_for_status()
        text = response.json()['candidates'][0]['content']['parts'][0]['text']
        return text.strip()
    except Exception as e:
        logger.error(f"LLM enrichment failed: {e}")
        return prompt  # fallback to original prompt

def download_model_manually(model_name="all-MiniLM-L6-v2", fallback_to_scikit=True):
    """Manually download a model or fall back to scikit-learn if download fails."""
    logger.info("Attempting manual model download...")
    cache_dir = Path(os.path.join(os.getcwd(), "model_cache"))
    cache_dir.mkdir(exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(
            repo_id=f"sentence-transformers/{model_name}", 
            cache_dir=str(cache_dir),
            local_files_only=False
        )
        logger.info(f"Model downloaded successfully to {model_path}")
        return SentenceTransformer(model_path)
    except Exception as e:
        logger.error(f"Manual download failed: {e}")
        
        # Fall back to scikit-learn as a last resort
        if fallback_to_scikit:
            logger.warning("Using scikit-learn TfidfVectorizer as fallback")
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            class SklearnFallbackModel:
                def __init__(self):
                    self.vectorizer = TfidfVectorizer()
                    self.is_fitted = False
                
                def encode(self, texts, show_progress_bar=False):
                    if not self.is_fitted:
                        self.vectorizer.fit(texts)
                        self.is_fitted = True
                    return self.vectorizer.transform(texts).toarray()
            
            return SklearnFallbackModel()
        else:
            raise

def setup_engine(json_path="shl_products.json"):
    """
    Set up the recommendation engine with data loading, embeddings creation, and Gemini integration.
    
    Args:
        json_path (str): Path to the JSON file containing assessment data
        
    Returns:
        tuple: (model, index, data, gemini_api_key)
    """
    try:
        # Try to get API key from Streamlit secrets first, then fall back to .env
        try:
            import streamlit as st
            gemini_api_key = st.secrets["GEMINI_API_KEY"]
        except (ImportError, KeyError):
            # Fall back to .env file for local development
            load_dotenv()
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables or Streamlit secrets. Query enrichment will be disabled.")
        
        # Load data
        logger.info("Loading data...")
        data = load_data(json_path)
        
        # Create embeddings with retry logic
        logger.info("Creating embeddings...")
        try:
            embeddings, model = create_embeddings(data)
        except HfHubHTTPError as e:
            if "429" in str(e):
                logger.warning("Rate limit hit, trying manual download...")
                model = download_model_manually()
                rich_descriptions = []
                for item in data:
                    rich_text = f"""
                    Assessment Type: {item['assessment_type']}
                    Test Type: {item['test_type']}
                    Job Levels: {item['job_levels']}
                    Description: {item['description']}
                    """
                    rich_descriptions.append(rich_text.strip())
                embeddings = model.encode(rich_descriptions, show_progress_bar=True)
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        index = create_faiss_index(embeddings)
        
        return model, index, data, gemini_api_key
        
    except Exception as e:
        logger.error(f"Error in setup_engine: {e}")
        raise

def main():
    try:
        api_key = os.getenv("GEMINI_API_KEY")  # make sure to export this in your terminal or .env
        logger.info("Loading data...")
        data = load_data("shl_products.json")
        
        logger.info("Creating embeddings...")
        embeddings, model = create_embeddings(data)
        
        logger.info("Creating FAISS index...")
        index = create_faiss_index(embeddings)

        # User interaction
        print("\nEnter your query (or press Ctrl+C to exit):")
        query = input("> ")

        # Step 1: Enrich query
        logger.info("Enhancing query using Gemini...")
        enhanced_query = enrich_query_with_gemini(query, api_key)
        print(f"\n🔍 Enhanced Query: {enhanced_query}")

        # Step 2: Get recommendations
        logger.info("Getting recommendations...")
        results = get_top_k_recommendations(enhanced_query, model, index, data)

        # Step 3: Show results
        print("\nTop Recommendations:")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['assessment_name']} ({r['assessment_type']}) - {r['url']}")
            print(f"   Remote: {r['remote_testing']}, Adaptive/IRT: {r['adaptive_irt']}")
            print(f"   Duration: {r['duration']}, Test Type: {r['test_type']}")
            print(f"   Description: {r['description'][:200]}...")

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

    try:
        # Load data
        logger.info("Loading data...")
        data = load_data("shl_products.json")
        
        # Create embeddings
        logger.info("Creating embeddings...")
        embeddings, model = create_embeddings(data)
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        index = create_faiss_index(embeddings)
        
        # Get user query
        print("\nEnter your query (or press Ctrl+C to exit):")
        query = input("> ")
        
        # Get and display recommendations
        logger.info("Getting recommendations...")
        results = get_top_k_recommendations(query, model, index, data)
        
        print("\nTop Recommendations:")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['assessment_name']} ({r['assessment_type']}) - {r['url']}")
            print(f"   Remote: {r['remote_testing']}, Adaptive/IRT: {r['adaptive_irt']}")
            print(f"   Duration: {r['duration']}, Test Type: {r['test_type']}")
            print(f"   Description: {r['description'][:200]}...")  # Show first 200 chars of description
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
