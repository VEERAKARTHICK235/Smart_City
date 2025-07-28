from transformers import pipeline

# Initialize the sentiment analysis pipeline once
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_feedback(text):
    """Analyzes sentiment of feedback and generates a summary."""
    # 1. Sentiment Analysis using a pre-trained model from Hugging Face
    sentiment = sentiment_pipeline(text)[0]
    label = sentiment['label']
    score = sentiment['score']
    
    # 2. LLM Simulation: Generate a simple summary based on sentiment
    summary = ""
    if label == 'POSITIVE':
        summary = f"Positive feedback received (Confidence: {score:.2f}). Suggestion: Acknowledge and thank the citizen."
    elif label == 'NEGATIVE':
        summary = f"Negative feedback received (Confidence: {score:.2f}). Suggestion: Escalate to the relevant department for review."
    else: # NEUTRAL
        summary = f"Neutral feedback received (Confidence: {score:.2f}). Suggestion: Log for informational purposes."
        
    return {"original_text": text, "sentiment": label, "summary": summary}

# Example Usage:
# feedback1 = "The new park is beautiful and clean. Great job!"
# feedback2 = "The traffic light on main street is broken again, causing huge delays."
# print(f"📣 {analyze_feedback(feedback1)}")
# print(f"📣 {analyze_feedback(feedback2)}")