from fastapi import FastAPI, Query
from engine import setup_engine, get_top_k_recommendations

app = FastAPI()
model, index, data = setup_engine()

@app.get("/recommend")
def recommend(query: str = Query(..., description="Job description or query")):
    results = get_top_k_recommendations(query, model, index, data)
    return results
