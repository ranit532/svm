import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plot_path = f'plots/confusion_matrix_{model_name}.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_decision_boundary(X, y, model, model_name):
    """Plots and saves the decision boundary of a model using PCA for visualization."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    model.fit(X_pca, y)

    h = .02  # step size in the mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Decision Boundary for {model_name}')
    plot_path = f'plots/decision_boundary_{model_name}.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def train_and_evaluate(data_path='data/synthetic_data.csv'):
    """
    Trains and evaluates SVM models with Linear and RBF kernels,
    logging results with MLflow.
    """
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models to train
    models = {
        'Linear_SVM': SVC(kernel='linear', C=1.0, random_state=42),
        'RBF_SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            print(f"Training {model_name}...")

            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log parameters
            mlflow.log_param("kernel", model.kernel)
            mlflow.log_param("C", model.C)
            if model.kernel == 'rbf':
                mlflow.log_param("gamma", model.gamma)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")

            # Log model
            mlflow.sklearn.log_model(model, model_name)

            # Generate and log plots
            cm_plot = plot_confusion_matrix(y_test, y_pred, model_name)
            mlflow.log_artifact(cm_plot, 'plots')
            
            # For decision boundary, we need a model trained on 2D data
            if X_train.shape[1] > 2:
                model_for_plot = SVC(kernel=model.kernel, C=model.C, gamma=model.gamma, random_state=42)
                db_plot = plot_decision_boundary(X_train.values, y_train.values, model_for_plot, model_name)
                mlflow.log_artifact(db_plot, 'plots')

            print(f"  Logged model, confusion matrix, and decision boundary for {model_name}.")

if __name__ == "__main__":
    # Set MLflow tracking URI if you have a server, otherwise it logs to local 'mlruns' directory
    # mlflow.set_tracking_uri("http://localhost:5000")
    
    # Set experiment
    mlflow.set_experiment("SVM_Classification_POC")
    
    train_and_evaluate()