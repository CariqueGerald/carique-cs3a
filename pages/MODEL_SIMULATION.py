import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class DataGenerator:
    def __init__(self):
        self.mean_values_dict = {}
        self.std_values_dict = {}
    
    def initialize_parameters(self, features, classes):
        """Initialize or update feature parameters for each class"""
        for class_name in classes:
            if class_name not in self.mean_values_dict:
                self.mean_values_dict[class_name] = [np.random.uniform(50, 150) for _ in features]
                self.std_values_dict[class_name] = [round(np.random.uniform(5.0, 15.0), 1) for _ in features]
            else:
                self._adjust_feature_count(class_name, features)

    def _adjust_feature_count(self, class_name, features):
        """Adjust feature count for existing classes"""
        current_count = len(self.mean_values_dict[class_name])
        target_count = len(features)
        
        if current_count < target_count:
            self.mean_values_dict[class_name].extend([np.random.uniform(50, 150) for _ in range(target_count - current_count)])
            self.std_values_dict[class_name].extend([round(np.random.uniform(5.0, 15.0), 1) for _ in range(target_count - current_count)])
        elif current_count > target_count:
            self.mean_values_dict[class_name] = self.mean_values_dict[class_name][:target_count]
            self.std_values_dict[class_name] = self.std_values_dict[class_name][:target_count]

    def generate_data(self, features, classes, total_samples):
        """Generate synthetic data for each class"""
        samples_per_class = total_samples // len(classes)
        remainder = total_samples % len(classes)
        
        generated_data = []
        for i, class_name in enumerate(classes):
            extra_samples = 1 if i < remainder else 0
            num_samples = samples_per_class + extra_samples
            
            data = np.random.normal(
                loc=self.mean_values_dict[class_name],
                scale=self.std_values_dict[class_name],
                size=(num_samples, len(features))
            )
            labels = np.full((num_samples, 1), class_name)
            generated_data.append(np.hstack([data, labels]))
            
        return np.vstack(generated_data)

class MLModelTrainer:
    def __init__(self):
        self.models = {
            "Random Forest": RandomForestClassifier(),
            "AdaBoost": AdaBoostClassifier(algorithm='SAMME'),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(),
            "Neural Network": MLPClassifier(max_iter=500),
            "Extra Trees": ExtraTreesClassifier()
        }
        self.results = {}
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        best_model = None
        best_score = 0
        
        for name, model in self.models.items():
            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                y_pred = model.predict(X_test)
                metrics = self._calculate_metrics(y_test, y_pred)
                metrics["Training Time"] = round(train_time, 4)
                metrics["Status"] = "Success"
                
                self.results[name] = metrics
                
                if metrics["Accuracy"] > best_score:
                    best_score = metrics["Accuracy"]
                    best_model = model
                    
            except Exception as e:
                self.results[name] = {
                    "Accuracy": None,
                    "Precision": None,
                    "Recall": None,
                    "F1-Score": None,
                    "Training Time": None,
                    "Status": f"Failed: {str(e)}"
                }
                
        return best_model
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted'),
            "Recall": recall_score(y_true, y_pred, average='weighted'),
            "F1-Score": f1_score(y_true, y_pred, average='weighted')
        }

class Visualizer:
    @staticmethod
    def plot_confusion_matrix(model, X_test, y_test, model_name, class_names, accuracy):
        """Generate confusion matrix plot"""
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f"{model_name}\nAccuracy: {accuracy:.2%}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        return fig
    
    @staticmethod
    def plot_learning_curve(estimator, title, X, y):
        """Generate learning curve plot"""
        train_sizes = np.linspace(0.1, 1.0, 5)
        plt.figure(figsize=(4, 4))
        plt.title(title)
        plt.xlabel("Training Examples")
        plt.ylabel("Score")
        
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, train_sizes=train_sizes, scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.grid()
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_mean - test_std,
                        test_mean + test_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        return plt

class StreamlitApp:
    def __init__(self):
        self.data_generator = DataGenerator()
        self.model_trainer = MLModelTrainer()
        self.visualizer = Visualizer()
        
    def run(self):
        st.title("🤖 ML Model Generator 🤖")
        
        # Sidebar inputs
        features = self._get_feature_input()
        classes = self._get_class_input()
        total_samples = st.sidebar.slider("Number of samples", 500, 50000, 1000, 500)
        train_split = st.sidebar.slider("Train-Test Split (%)", 10, 50, 20, 5)
        
        if st.sidebar.button("Generate Data and Train Models"):
            self._process_data(features, classes, total_samples, train_split)
    
    def _get_feature_input(self):
        feature_input = st.sidebar.text_input(
            "Enter feature names (comma-separated)",
            "length (mm), width (mm), density (g/cm³)"
        )
        return [f.strip() for f in feature_input.split(",")]
    
    def _get_class_input(self):
        class_input = st.sidebar.text_input(
            "Enter class names (comma-separated)",
            "Ampalaya, Banana, Cabbage"
        )
        return [c.strip() for c in class_input.split(",")]
    
    def _process_data(self, features, classes, total_samples, train_split):
        # Generate data
        self.data_generator.initialize_parameters(features, classes)
        generated_data = self.data_generator.generate_data(features, classes, total_samples)
        
        # Create and display DataFrames
        X = generated_data[:, :-1].astype(float)
        y = generated_data[:, -1]
        df = pd.DataFrame(X, columns=features)
        df['Target'] = y
        
        self._display_data_info(df, train_split, total_samples)
        self._display_visualizations(df, features)
        
        # Train and evaluate models
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=train_split/100, random_state=42
        )
        
        best_model = self.model_trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        if best_model:
            self._display_model_results(best_model, X_test, y_test, classes)
            self._save_and_offer_downloads(df, self.model_trainer.models, self.model_trainer.results)
            
    def _display_data_info(self, df, train_split, total_samples):
        st.subheader("📊 Dataset Overview")
        st.dataframe(df.head())
        
        train_samples = int(total_samples * (1 - train_split/100))
        test_samples = total_samples - train_samples
        
        cols = st.columns(3)
        cols[0].metric("Total Samples", total_samples)
        cols[1].metric("Training Samples", train_samples)
        cols[2].metric("Testing Samples", test_samples)
    
    def _display_visualizations(self, df, features):
        st.subheader("📈 Feature Visualization")
        viz_type = st.radio("Select Visualization", ["2D", "3D"])
        
        if viz_type == "2D":
            x_feat = st.selectbox("X-axis feature", features, 0)
            y_feat = st.selectbox("Y-axis feature", features, 1)
            fig = px.scatter(df, x=x_feat, y=y_feat, color="Target")
            st.plotly_chart(fig)
        else:
            x_feat = st.selectbox("X-axis feature", features, 0)
            y_feat = st.selectbox("Y-axis feature", features, 1)
            z_feat = st.selectbox("Z-axis feature", features, 2)
            fig = px.scatter_3d(df, x=x_feat, y=y_feat, z=z_feat, color="Target")
            st.plotly_chart(fig)
    
    def _display_model_results(self, best_model, X_test, y_test, classes):
        st.subheader("🎯 Model Performance")
        
        # Display results table
        results_df = pd.DataFrame(self.model_trainer.results).T
        st.dataframe(results_df)
        
        # Display confusion matrices
        st.subheader("Confusion Matrices")
        cols = st.columns(3)
        for i, (name, model) in enumerate(self.model_trainer.models.items()):
            if self.model_trainer.results[name]["Status"] == "Success":
                fig = self.visualizer.plot_confusion_matrix(
                    model, X_test, y_test, name, classes,
                    self.model_trainer.results[name]["Accuracy"]
                )
                cols[i % 3].pyplot(fig)
    
    def _save_and_offer_downloads(self, df, models, results):
        st.subheader("💾 Download Options")
        
        # Save dataset
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Dataset (CSV)",
            csv,
            "synthetic_dataset.csv",
            "text/csv"
        )
        
        # Save models
        os.makedirs("saved_models", exist_ok=True)
        for name, model in models.items():
            if results[name]["Status"] == "Success":
                joblib.dump(model, f"saved_models/{name}.pkl")
                
                with open(f"saved_models/{name}.pkl", "rb") as f:
                    st.download_button(
                        f"Download {name} Model",
                        f,
                        f"{name}_model.pkl",
                        "application/octet-stream"
                    )

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
