import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("📩 Spam Message Classifier")

st.write("Enter a message to check whether it is Spam or Not Spam")

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', names=["label", "message"])

# Convert labels to numbers
data['label'] = data.label.map({'ham': 0, 'spam': 1})

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2
)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_counts, y_train)

# User input
user_input = st.text_input("Type your message here:")

# Button
if st.button("Check"):
    msg = [user_input]
    msg_count = vectorizer.transform(msg)
    prediction = model.predict(msg_count)

    if prediction[0] == 1:
        st.error("🚨 This is a Spam Message")
    else:
        st.success("✅ This is Not a Spam Message")
