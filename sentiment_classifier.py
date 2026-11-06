positive_reviews = [
    "excellent service and fast response",
    "great support and helpful staff",
    "very happy with this product",
    "the app is easy and useful",
    "good quality and smooth experience",
    "quick delivery and perfect order",
    "friendly team and excellent help",
    "love this simple and reliable app",
    "the update is much better",
    "very satisfied with the service",
    "amazing experience and great quality",
    "support was quick and helpful",
    "everything worked perfectly",
    "easy setup and good performance",
    "the product is reliable and useful",
    "fast service and friendly support",
    "really good experience overall",
    "happy with the quick response",
    "great product and excellent quality",
    "simple interface and smooth performance",
]

negative_reviews = [
    "terrible service and slow response",
    "poor support and unhelpful staff",
    "very unhappy with this product",
    "the app is difficult and useless",
    "bad quality and frustrating experience",
    "late delivery and wrong order",
    "rude team and terrible help",
    "hate this confusing and unreliable app",
    "the update is much worse",
    "very disappointed with the service",
    "awful experience and poor quality",
    "support was slow and useless",
    "nothing worked correctly",
    "difficult setup and bad performance",
    "the product is unreliable and frustrating",
    "slow service and rude support",
    "really bad experience overall",
    "unhappy with the slow response",
    "poor product and terrible quality",
    "confusing interface and frustrating performance",
]

texts = positive_reviews + negative_reviews
labels = ["positive"] * len(positive_reviews) + ["negative"] * len(negative_reviews)

print("Training examples:", len(texts))
print("Labels:", len(labels))

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
        ),
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000),
    ),
])

model.fit(texts, labels)

test_reviews = [
    "the service was fast and helpful",
    "this product is excellent and reliable",
    "the app is confusing and difficult",
    "delivery was late and the order was wrong",
]

predictions = model.predict(test_reviews)

for review, prediction in zip(test_reviews, predictions):
    print("\nReview:", review)
    print("Prediction:", prediction)

probabilities = model.predict_proba(test_reviews)
classes = model.classes_

for review, prediction, scores in zip(
    test_reviews,
    predictions,
    probabilities,
):
    print("\nReview:", review)
    print("Prediction:", prediction)

    for class_name, score in zip(classes, scores):
        print(class_name, "=", round(score, 3))

tfidf = model.named_steps["tfidf"]
feature_names = tfidf.get_feature_names_out()

print("\nNumber of text features:", len(feature_names))
print("First 20 features:")
print(feature_names[:20])

classifier = model.named_steps["classifier"]
weights = classifier.coef_[0]

feature_weight_pairs = list(zip(feature_names, weights))

most_negative = sorted(feature_weight_pairs, key=lambda item: item[1])[:10]
most_positive = sorted(feature_weight_pairs, key=lambda item: item[1], reverse=True)[:10]

print("\nFeatures associated with negative sentiment:")
for feature, weight in most_negative:
    print(feature, round(weight, 3))

print("\nFeatures associated with positive sentiment:")
for feature, weight in most_positive:
    print(feature, round(weight, 3))

tricky_reviews = [
    "meh, it works",
    "the support response was insanely quick",
    "chala bagundi",
    "parledu",
    "baledu",
    "yemi baledu",
]

tricky_predictions = model.predict(tricky_reviews)

for review, prediction in zip(tricky_reviews, tricky_predictions):
    print("\nReview:", review)
    print("Prediction:", prediction)

