from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def generate_recommendations(input_text):
    """
    Generate recommendations using GPT-2.
    """
    # Load the GPT-2 model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Tokenize input and generate output
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    
    # Decode and return the result
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def analyze_sentiment(reviews):
    """
    Analyze sentiment using a BERT-based model.
    """
    # Load a sentiment-analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis")
    
    # Analyze each review and return results
    results = []
    for review in reviews:
        result = sentiment_analyzer(review)
        results.append(result[0])  # Get the first result
    return results
