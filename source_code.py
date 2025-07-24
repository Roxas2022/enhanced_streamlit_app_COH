import streamlit as st
from sklearn.datasets import fetch_20newsgroups
# news_classifier_app.py

import streamlit as st  # Streamlit library for building web apps with Python
from sklearn.datasets import fetch_20newsgroups  # Loads the 20 Newsgroups text classification dataset
from sklearn.pipeline import Pipeline  # Allows chaining preprocessing and model steps into one object
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # Text preprocessing: CountVectorizer converts text to token counts; TfidfTransformer scales those counts by importance
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier for multinomially distributed data (good for text)
from sklearn.linear_model import SGDClassifier  # Support Vector Machine (SVM) using Stochastic Gradient Descent (fast linear classifier)
from sklearn.ensemble import RandomForestClassifier  # Ensemble learning method that combines multiple decision trees (Random Forest)
from sklearn.tree import DecisionTreeClassifier  # A simple decision tree classifier for supervised learning
import numpy as np  # NumPy for numerical operations like calculating mean (accuracy)


# Title and intro
st.title("News Classifier App, (AKA, enhancing a streamlit app)")
st.write("This app uses these models for classification: Naïve Bayes, SVM, Random Forest, Decision Tree.")
st.write("Coded and brought to you by Carlos O Hulse")
st.write("CAI2300C")
st.write("7/23/2025")
st.write("Professor Muturi")

# Sidebar classifier options
cs = ["Naive Bayes", "SVM", "Random Forest", "Decision Tree"]
classification_space = st.sidebar.selectbox("Choose Classifier", cs)

# Trigger classification
if st.sidebar.button("Classify"):
    # Load training and test data
    trainData = fetch_20newsgroups(subset='train', shuffle=True)
    test_set = fetch_20newsgroups(subset='test', shuffle=True)

    # Select model
    if classification_space == "Naive Bayes":
        st.write("Naive Bayes selected")
        classifier = MultinomialNB()
    
    elif classification_space == "SVM":
        st.write("SVM selected")
        classifier = SGDClassifier(loss='hinge', penalty='l1', alpha=0.0005, l1_ratio=0.17)
    
    elif classification_space == "Random Forest":
        st.write("Random Forest selected")
        classifier = RandomForestClassifier(max_depth=2, random_state=0)

    elif classification_space == "Decision Tree":
        st.write("Decision Tree selected")
        classifier = DecisionTreeClassifier(random_state=0)

    # Create and train pipeline
    pipeline = Pipeline([
        ('bow', CountVectorizer()),
        ('vector', TfidfTransformer()),
        ('classifier', classifier)
    ])

    pipeline.fit(trainData.data, trainData.target)
    predictions = pipeline.predict(test_set.data)

    # Show accuracy
    accuracy = np.mean(predictions == test_set.target)
    st.write("Model Accuracy:")
    st.write(f"{accuracy:.4f}")
