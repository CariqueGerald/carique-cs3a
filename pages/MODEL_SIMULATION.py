import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random

# Function to generate synthetic data based on user input
def generate_synthetic_data(classes, features, num_samples):
    # Create a dictionary for original data
    original_data = {class_name: random.sample(features, len(features)) for class_name in classes}
    
    # Now we will generate synthetic data. 
    # For simplicity, we will use make_classification to simulate feature data
    X, y = make_classification(n_samples=num_samples, n_features=len(features), random_state=42)
    
    # Create DataFrame for synthetic data
    synthetic_data = pd.DataFrame(X, columns=features)
    synthetic_data['Class'] = np.random.choice(classes, num_samples)
    
    return original_data, synthetic_data

# Streamlit Sidebar Inputs
st.sidebar.title("Data Parameters")

# Step 1: Enter class names
class_names_input = st.sidebar.text_input("Enter class names separated by commas:", "banana, apple, carrot, orange")
class_names = [name.strip() for name in class_names_input.split(",")]

# Step 2: Enter features (including numbers)
features_input = st.sidebar.text_input("Enter features separated by commas:", "yellow, red, sweet, sour, 7, 5, 2")
features = [feature.strip() for feature in features_input.split(",")]

# Step 3: User input for assigning features to classes
st.sidebar.subheader("Class Specific Parameters")

# Collect user feature selection for each class
class_features = {}
for class_name in class_names:
    selected_features = st.sidebar.multiselect(f"Specific features for {class_name}", options=features)
    class_features[class_name] = selected_features

# Step 4: User input for generating synthetic data (number of samples)
num_samples = st.sidebar.slider("Select number of synthetic data samples", 1000, 10000, 5000)

# Step 5: Button to generate synthetic data
generate_button = st.sidebar.button("Generate Synthetic Data")

if generate_button:
    if class_names and features and class_features:
        # Generate original data and synthetic data
        original_data, synthetic_data = generate_synthetic_data(class_names, features, num_samples)

        # Display the original data and synthetic data
        st.subheader("Original Data")
        st.write(original_data)
        
        st.subheader(f"Synthetic Data (Generated {num_samples} samples)")
        st.write(synthetic_data)
    else:
        st.sidebar.error("Please enter class names and features properly.")

# Main content of the app
st.title("Synthetic Data Generation")

st.write(
    "This app allows you to generate synthetic data based on the user's original input of class names and features."
)

