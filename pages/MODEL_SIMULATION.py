import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px

def generate_synthetic_data(features, classes, total_samples):
    samples_per_class = total_samples // len(classes)
    data = []
    for cls in classes:
        class_data = np.random.normal(loc=0, scale=1, size=(samples_per_class, len(features)))
        labels = np.full((samples_per_class, 1), cls)
        data.append(np.hstack((class_data, labels)))
    return np.vstack(data), features + ["Target"]

def main():
    st.title("Simple Modeling and Simulation App")

    # Sidebar settings
    st.sidebar.header("Synthetic Data Settings")
    features = st.sidebar.text_input("Enter feature names (comma-separated):", "Feature1, Feature2, Feature3").split(",")
    classes = st.sidebar.text_input("Enter class names (comma-separated):", "Class1, Class2").split(",")
    total_samples = st.sidebar.slider("Number of samples", min_value=1000, max_value=10000, step=500)
    train_percent = st.sidebar.slider("Train-Test Split (%)", min_value=10, max_value=90, step=5)

    if st.sidebar.button("Generate and Train"):
        data, columns = generate_synthetic_data(features, classes, total_samples)
        df = pd.DataFrame(data, columns=columns)

        X = df[features].astype(float)
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_percent) / 100, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display results
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Evaluation Metrics")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        st.subheader("2D Scatter Plot")
        fig = px.scatter(df, x=features[0], y=features[1], color="Target", title="Synthetic Data Visualization")
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
