import gradio as gr
import pandas as pd

# Load data
segmented_data = pd.read_csv("./output_data/segmented_data.csv")
sentiment_data = pd.read_csv("./output_data/sentiment_analysis.csv")

# Define a function to display segmented data
def show_segmented_data():
    return segmented_data

# Define a function to display sentiment analysis data
def show_sentiment_analysis():
    return sentiment_data

# Create a Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Customer Segmentation and Sentiment Analysis")
    
    gr.Markdown("### Segmented Data")
    gr.Dataframe(segmented_data, label="Segmented Data", interactive=True)

    gr.Markdown("### Sentiment Analysis")
    gr.Dataframe(sentiment_data, label="Sentiment Analysis", interactive=True)

# Launch the app
demo.launch()
