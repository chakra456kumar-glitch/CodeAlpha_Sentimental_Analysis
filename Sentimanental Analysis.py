# Load dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



df = pd.read_csv("amazon_reviews.csv")

print(df.head())

# Clean Data

df = df.dropna()

# Convert comments to lowercase

df['Comment'] = df['Comment'].str.lower()

# Sentiment Classification using Score column

def get_sentiment(score):

    if score == 1:
        return 'Negative'

    elif score == 2:
        return 'Positive'

    else:
        return 'Neutral'

# Create Sentiment column
df['Sentiment'] = df['Score'].apply(get_sentiment)

# Count sentiments
print(df['Sentiment'].value_counts())

# Pie Chart

df['Sentiment'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)

plt.title("Sentiment Distribution")
plt.ylabel("")
plt.show()

# Bar Chart

sns.countplot(x='Sentiment', data=df)

plt.title("Sentiment Analysis")
plt.show()

# Generate Word Cloud
# Create text from comments
df = df.head(5000)
text = " ".join(df['Comment'].astype(str))


wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(text)

# Create plot window
plt.figure(figsize=(12,6))

# Show Word Cloud
plt.imshow(wordcloud)

# Remove axis
plt.axis("off")

# Add title
plt.title("Word Cloud of Amazon Reviews")

# Display output
plt.show(block=True)

# MACHINE LEARNING MODELS

X = df['Comment']
y = df['Sentiment']

# Convert text into numerical data
vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)


# LOGISTIC REGRESSION MODEL

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

# Accuracy
lr_accuracy = accuracy_score(y_test, lr_predictions)

print("\nLogistic Regression Accuracy:")
print(lr_accuracy)

# Report
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))


#Project includes:

#NLP
#Data Visualization
#Machine Learning
#Model Evaluation

#Understanding Each Metric
#Accuracy percentage--accuracy = 0.80--our model correctly predicted 80% of the reviews.
#Out of 1000 test reviews:About 800 predictions were correct, About 200 predictions were incorrect

#Metric	   Meaning

#Precision  How many predicted labels were actually correct
#Recall	    How many actual labels were correctly found
#F1-score   Balance between precision and recall
#Support	   Number of actual samples








