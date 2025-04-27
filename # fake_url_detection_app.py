# fake_url_detection_app.py

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# -----------------------------------
# Step 1: Feature extraction function
# -----------------------------------
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['has_at_symbol'] = 1 if '@' in url else 0
    features['count_dots'] = url.count('.')
    features['has_ip_address'] = 1 if re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', url) else 0
    features['has_suspicious_words'] = 1 if any(word in url.lower() for word in ['login', 'secure', 'account', 'update', 'free', 'verify']) else 0
    return features

# -----------------------------------
# Step 2: Prepare Dataset
# -----------------------------------
def prepare_dataset():
    # Example dataset (you can replace with a real dataset later)
    data = {
        'url': [
            'http://www.google.com',
            'https://secure-login.com/login',
            'http://192.168.0.1/secure',
            'https://bankofamerica.com.login.verify-secure.com',
            'http://normalwebsite.com/page',
            'http://badwebsite.com/@login',
            'https://secure.paypal.com.fake-site.com',
            'http://free-gift-cards.com',
            'https://www.amazon.com',
            'http://verify-paypal-login.com'
        ],
        'label': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0]  # 1 = Legit, 0 = Fake
    }
    df = pd.DataFrame(data)
    feature_list = [extract_features(url) for url in df['url']]
    features_df = pd.DataFrame(feature_list)
    X = features_df
    y = df['label']
    return X, y

# -----------------------------------
# Step 3: Train the Model
# -----------------------------------
def train_model():
    X, y = prepare_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc

# -----------------------------------
# Step 4: Streamlit Frontend
# -----------------------------------
def main():
    st.set_page_config(page_title="Fake URL Detection", layout="centered")
    st.title("🔍 Fake URL Detection App")
    st.write("Predict whether a URL is **Legitimate** or **Fake/Phishing** using Machine Learning!")

    # Train model
    model, accuracy = train_model()

    # URL Input
    url_input = st.text_input("Enter URL to check:")

    if st.button("Predict"):
        if url_input:
            features = extract_features(url_input)
            features_values = np.array(list(features.values())).reshape(1, -1)
            prediction = model.predict(features_values)
            if prediction[0] == 1:
                st.success("✅ This URL seems Legitimate!")
            else:
                st.error("❌ Warning! This URL seems Fake!")
        else:
            st.warning("Please enter a URL first!")

    # Display model accuracy
    if st.checkbox("Show Model Accuracy"):
        st.info(f"Model Accuracy: {accuracy*100:.2f}%")

    # Display decision tree visualization
    if st.checkbox("Show Decision Tree Visualization"):
        X, _ = prepare_dataset()
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(model, filled=True, feature_names=X.columns, class_names=["Fake", "Legit"])
        st.pyplot(fig)

if __name__ == "__main__":
    main()
