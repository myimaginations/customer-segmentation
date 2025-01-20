import pandas as pd

def generate_recommendations(segmented_data, sentiment_data):
    """
    Generate dynamic recommendations based on customer segmentation and sentiment analysis.
    """
    # Merge sentiment data with segmented data
    sentiment_data = sentiment_data.rename(columns={"Review": "CustomerFeedback"})
    data = pd.merge(segmented_data, sentiment_data, left_index=True, right_index=True, how='left')

    # Add recommendation column
    def recommend(row):
        if row['Cluster'] == 0 and row['Sentiment'] == 'POSITIVE':
            return "Offer Premium Products"
        elif row['Cluster'] == 1 and row['Sentiment'] == 'NEGATIVE':
            return "Offer Discounts or Support"
        elif row['Cluster'] == 2:
            return "Promote New Arrivals"
        else:
            return "General Promotions"

    data['Recommendation'] = data.apply(recommend, axis=1)

    return data

