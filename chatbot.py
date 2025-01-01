import streamlit as st
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk

# Load the intents file
with open('intents.json') as f:
    intents = json.load(f)

# Prepare data for model training
patterns = []
tags = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Train a simple logistic regression model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
model = LogisticRegression()
model.fit(X, tags)

# Define the chatbot's response function
def chatbot_response(user_input):
    input_vector = vectorizer.transform([user_input])
    predicted_tag = model.predict(input_vector)[0]

    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I don't understand."

# Streamlit UI
st.title("Healthcare Chatbot")
st.write("Type your health-related queries below and hit Enter.")

user_input = st.text_input("You:", "")

if user_input:
    response = chatbot_response(user_input)
    st.write(f"Chatbot: {response}")