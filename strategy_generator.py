import openai

def generate_cluster_strategies(segmented_data):
    """
    Generate marketing strategies for each cluster using GPT models.
    """
    # Define the OpenAI API key
    openai.api_key = "YOUR_API_KEY_HERE"

    cluster_strategies = {}
    for cluster in segmented_data['Cluster'].unique():
        try:
            # Prepare the prompt
            prompt = f"Generate a marketing strategy for customer cluster {cluster} based on their behavior and preferences."

            # Use the ChatCompletion API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a marketing expert."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract the strategy from the response
            strategy = response['choices'][0]['message']['content']
            cluster_strategies[cluster] = strategy
        except Exception as e:
            # Handle any errors
            cluster_strategies[cluster] = f"Error generating strategy for Cluster {cluster}: {str(e)}"

    return cluster_strategies
