import nltk
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt_tab')

# Configure NLTK data path
nltk.data.path.clear()
nltk.data.path.append('./Backend/nltk_data')
nltk.download('punkt_tab', download_dir='./Backend/nltk_data')

# --- Cleaning utility ---
def clean_text(raw_text):
    cleaned = re.sub(r'\[[0-9]*\]', '', raw_text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

# --- Core summarization logic ---
def summarize_content(content, max_lines=3):
    refined_text = clean_text(content)
    segments = nltk.sent_tokenize(refined_text)

    if len(segments) <= max_lines:
        return "Provided text is too short to generate a summary."

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_vectors = tfidf.fit_transform(segments)

    similarity_scores = cosine_similarity(tfidf_vectors)
    relevance = similarity_scores.sum(axis=1)

    top_indices = np.argsort(relevance)[-max_lines:]
    selected_sentences = [segments[i] for i in sorted(top_indices)]

    return ' '.join(selected_sentences)

# --- Interaction layer ---
def main():
    print("Paste your article or paragraph to get a summary:")
    user_text = input(">> ")

    result = summarize_content(user_text)
    print("\n--- Summary ---")
    print(result)

# --- Entry point ---
if __name__ == "__main__":
    main()
