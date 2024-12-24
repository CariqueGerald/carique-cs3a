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

def plot_2d_scatter(df, x_feature, y_feature):
    """Create 2D scatter plot with selected features."""
    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color="Target",
        title=f"{x_feature} vs {y_feature}",
        labels={x_feature: x_feature, y_feature: y_feature}
    )
    return fig

def plot_3d_scatter(df, x_feature, y_feature, z_feature):
    """Create 3D scatter plot with selected features."""
    fig = px.scatter_3d(
        df,
        x=x_feature,
        y=y_feature,
        z=z_feature,
        color="Target",
        title=f"3D Plot: {x_feature}, {y_feature}, {z_feature}",
        labels={
            x_feature: x_feature,
            y_feature: y_feature,
            z_feature: z_feature
        }
    )
    return fig

def create_visualizations(df, features):
    """Create interactive data visualizations."""
    st.subheader("Data Visualization")
    
    viz_type = st.radio("Select Visualization Type", ["2D", "3D"])
    
    if viz_type == "2D":
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Select X-axis Feature", features)
        with col2:
            y_feature = st.selectbox("Select Y-axis Feature", features)
        
        fig = plot_2d_scatter(df, x_feature, y_feature)
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # 3D visualization
        col1, col2, col3 = st.columns(3)
        with col1:
            x_feature = st.selectbox("Select X-axis Feature", features)
        with col2:
            y_feature = st.selectbox("Select Y-axis Feature", features)
        with col3:
            z_feature = st.selectbox("Select Z-axis Feature", features)
        
        fig = plot_3d_scatter(df, x_feature, y_feature, z_feature)
        st.plotly_chart(fig, use_container_width=True)

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

def plot_model_comparison(results):
    """Create visual comparison of model performances."""
    fig = px.bar(
        x=list(results.keys()),
        y=list(results.values()),
        labels={"x": "Model", "y": "Accuracy"},
        title="Model Performance Comparison"
    )
    fig.update_traces(marker_color='rgb(55, 83, 109)')
    return fig

def main():
    st.title("ML Model Training App")
    
    # Data Generation Parameters
    st.sidebar.header("Data Parameters")
    features = st.sidebar.text_input("Features (comma-separated)", "length,width,height").split(",")
    features = [f.strip() for f in features]  # Clean whitespace
    classes = st.sidebar.text_input("Classes (comma-separated)", "A,B,C").split(",")
    classes = [c.strip() for c in classes]
    sample_size = st.sidebar.slider("Sample Size", 100, 1000, 500)
    
    if st.sidebar.button("Generate Data"):
        # Generate and prepare data
        class_data = generate_synthetic_data(features, classes, sample_size)
        all_data = np.vstack(class_data)
        
        # Create original DataFrame
        df_original = pd.DataFrame(all_data, columns=features + ["Target"])
        
        # Create scaled DataFrame
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_original[features])
        df_scaled = pd.DataFrame(scaled_features, columns=features)
        df_scaled["Target"] = df_original["Target"]
        
        # Display original and scaled data
        st.subheader("Dataset Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Data")
            st.dataframe(df_original.head())
        with col2:
            st.write("Scaled Data")
            st.dataframe(df_scaled.head())
        
        # Download buttons
        st.subheader("Download Data")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Original Dataset",
                df_original.to_csv(index=False),
                "original_data.csv",
                "text/csv"
            )
        with col2:
            st.download_button(
                "Download Scaled Dataset",
                df_scaled.to_csv(index=False),
                "scaled_data.csv",
                "text/csv"
            )
        
        # Create visualizations
        create_visualizations(df_original, features)
        
        # Prepare for model training
        X = df_scaled[features]
        y = df_scaled["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train and evaluate models
        best_model, results, models = train_and_evaluate_models(
            X_train, X_test, y_train, y_test
        )
        
        # Display results
        st.subheader("Model Performance")
        
        # Text results
        for model_name, accuracy in results.items():
            st.write(f"{model_name}: {accuracy:.2%}")
        
        # Visual comparison
        fig = plot_model_comparison(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Save best model
        if st.button("Save Best Model"):
            os.makedirs("models", exist_ok=True)
            joblib.dump(best_model, "models/best_model.pkl")
            joblib.dump(scaler, "models/scaler.pkl")
            st.success("Model and scaler saved successfully!")

if __name__ == "__main__":
    main()
