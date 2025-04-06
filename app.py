import gradio as gr
from engine import setup_engine, get_top_k_recommendations
import pandas as pd

# Load the model and index once at the start
model, index, data = setup_engine()

def recommend(query, top_k):
    if not query.strip():
        return pd.DataFrame()

    results = get_top_k_recommendations(query, model, index, data)
    top_results = results[:top_k]

    df = pd.DataFrame(top_results)[[
        'assessment_name', 'url', 'remote_testing', 'adaptive_irt', 'duration', 'test_type'
    ]]
    df.columns = [
        "Assessment Name", "URL", "Remote Testing", "Adaptive/IRT", "Duration", "Test Type"
    ]
    df["Assessment Name"] = df.apply(
        lambda row: f"[{row['Assessment Name']}]({row['URL']})", axis=1)
    df.drop(columns=["URL"], inplace=True)
    return df

with gr.Blocks(title="SHL Test Recommender") as demo:
    gr.Markdown("## 🔍 SHL Semantic Test Recommender")
    query_input = gr.Textbox(label="Enter your job description or requirement:")
    top_k_radio = gr.Radio([1, 3, 5, 10], label="Top K Recommendations", value=5)
    recommend_button = gr.Button("Recommend")
    output_table = gr.Dataframe(label="Recommendations", wrap=True)

    recommend_button.click(
        recommend,
        inputs=[query_input, top_k_radio],
        outputs=output_table
    )

demo.launch()
