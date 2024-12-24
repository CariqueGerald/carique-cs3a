import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import plotly.express as px
import joblib
import os

def generate_synthetic_data(features, classes, sample_size):
    """Generate synthetic data based on user parameters."""
    samples_per_class = sample_size // len(classes)
    class_data = []
    
    for class_name in classes:
        mean_values = np.random.uniform(50, 150, len(features))
        std_values = np.random.uniform(5, 15, len(features))
        
        data = np.random.normal(
            loc=mean_values,
            scale=std_values,
            size=(samples_per_class, len(features))
        )
        labels = np.full((samples_per_class, 1), class_name)
        class_data.append(np.hstack([data, labels]))
    
    return class_data

def create_visualizations(df, features):
    """Create basic data visualizations."""
    st.subheader("Data Visualization")
    
    # 2D Scatter Plot
    if len(features) >= 2:
        fig = px.scatter(
            df,
            x=features[0],
            y=features[1],
            color="Target",
            title="Feature Distribution"
        )
        st.plotly_chart(fig)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models."""
    models = {
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        results[name] = score
        
        if score > best_score:
            best_score = score
            best_model = model
    
    return best_model, results, models

def save_model(model, model_name):
    """Save trained model to disk."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}.pkl")

def main():
    st.title("ML Model Training App")
    
    # Data Generation Parameters
    st.sidebar.header("Data Parameters")
    features = st.sidebar.text_input("Features (comma-separated)", "length,width,height").split(",")
    classes = st.sidebar.text_input("Classes (comma-separated)", "A,B,C").split(",")
    sample_size = st.sidebar.slider("Sample Size", 100, 1000, 500)
    
    if st.sidebar.button("Generate Data"):
        # Generate and prepare data
        class_data = generate_synthetic_data(features, classes, sample_size)
        all_data = np.vstack(class_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_data, columns=features + ["Target"])
        st.write("Generated Data Sample:", df.head())
        
        # Create visualizations
        create_visualizations(df, features)
        
        # Prepare for model training
        X = df[features].astype(float)
        y = df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate models
        best_model, results, models = train_and_evaluate_models(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Display results
        st.subheader("Model Performance")
        for model_name, accuracy in results.items():
            st.write(f"{model_name}: {accuracy:.2%}")
        
        # Save best model
        if st.button("Save Best Model"):
            save_model(best_model, "best_model")
            st.success("Model saved successfully!")

if __name__ == "__main__":
    main()
