import re

text = "the delivery is very late"

normalized_text = text.lower()

tokens = re.findall(r"\b\w+\b", normalized_text)

print("Original:", text)
print("Normalized:", normalized_text)
print("Tokens:", tokens)

from sklearn.feature_extraction.text import CountVectorizer


sentences = [
    "good service",
    "good product",
    "bad service",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("\nNumeric representation:")
print(X.toarray())

from sklearn.feature_extraction.text import TfidfVectorizer


reviews = [
    "fast delivery and excellent service",
    "slow delivery and terrible service",
    "excellent product and great support",
]

tfidf = TfidfVectorizer()
review_vectors = tfidf.fit_transform(reviews)

print("TF-IDF features:")
print(tfidf.get_feature_names_out())

print("\nVector shape:")
print(review_vectors.shape)

print("\nTF-IDF values:")
print(review_vectors.toarray().round(2))
