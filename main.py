from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.clustering import apply_clustering
from src.huggingface_integration import generate_recommendations, analyze_sentiment
import pandas as pd

# File paths
DATA_PATH = './input_data/data.csv'
SEGMENTED_DATA_PATH = './output_data/segmented_data.csv'
STRATEGIES_PATH = './output_data/cluster_strategies.csv'
SENTIMENT_PATH = './output_data/sentiment_analysis.csv'
RECOMMENDATIONS_PATH = './output_data/recommendations.csv'

# Step 1: Load and preprocess data
data = load_and_clean_data(DATA_PATH)
print("Data after cleaning:")
print(data.info())

# Step 2: Feature engineering
features = create_features(data)
print("Features created and scaled.")

# Step 3: Clustering
segmented_data = apply_clustering(features)
segmented_data.to_csv(SEGMENTED_DATA_PATH, index=False)
print(f"Segmented data saved to {SEGMENTED_DATA_PATH}")

# Step 4: Generate strategies for clusters using GPT
strategies = {}
for cluster in segmented_data['Cluster'].unique():
    input_text = f"Based on customer cluster {cluster}, suggest strategies to improve engagement and retention."
    strategy = generate_recommendations(input_text)
    strategies[cluster] = strategy

# Save strategies to a CSV file
strategies_df = pd.DataFrame.from_dict(strategies, orient='index', columns=['Strategy'])
strategies_df.to_csv(STRATEGIES_PATH, index=True)
print(f"Cluster strategies saved to {STRATEGIES_PATH}")

# Step 5: Sentiment analysis of customer feedback
# Example reviews (replace with actual feedback if available)
reviews = [
    "The product is amazing!",
    "I'm unhappy with the delay in delivery.",
    "The service was average, but acceptable."
]
sentiments = analyze_sentiment(reviews)

# Save sentiment analysis results to a CSV file
sentiment_df = pd.DataFrame({
    "Review": reviews,
    "Sentiment": [s["label"] for s in sentiments],
    "Confidence": [s["score"] for s in sentiments]
})
sentiment_df.to_csv(SENTIMENT_PATH, index=False)
print(f"Sentiment analysis results saved to {SENTIMENT_PATH}")

# Step 6: Generate dynamic recommendations
def generate_dynamic_recommendations(segmented_data, sentiment_df):
    sentiment_df = sentiment_df.rename(columns={"Review": "CustomerFeedback"})
    merged_data = pd.merge(segmented_data, sentiment_df, left_index=True, right_index=True, how='left')

    def recommend(row):
        if row['Cluster'] == 0 and row['Sentiment'] == 'POSITIVE':
            return "Offer Premium Products"
        elif row['Cluster'] == 1 and row['Sentiment'] == 'NEGATIVE':
            return "Offer Discounts or Support"
        elif row['Cluster'] == 2:
            return "Promote New Arrivals"
        else:
            return "General Promotions"

    merged_data['Recommendation'] = merged_data.apply(recommend, axis=1)
    return merged_data

# Generate dynamic recommendations and save them
recommendations = generate_dynamic_recommendations(segmented_data, sentiment_df)
recommendations.to_csv(RECOMMENDATIONS_PATH, index=False)
print(f"Recommendations saved to {RECOMMENDATIONS_PATH}")
