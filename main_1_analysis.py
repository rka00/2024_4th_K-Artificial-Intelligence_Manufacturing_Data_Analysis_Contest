import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from lightgbm import plot_importance
from utils import *

def load_and_preprocess_data():
    # Load and preprocess data
    full_path = os.getcwd()
    data_path = os.path.join(full_path, 'data', '경진대회용 주조 공정최적화 데이터셋.csv')
    data = pd.read_csv(data_path, encoding='cp949')

    # Preprocess
    data = preprocess(data)
    data = make_time_series(data, time_threshold=3000)
    data = preprocess_time_series(data)
    data = make_dataframe(data, time_interval=30)

    # Train-valid-test split
    X_train, X_valid, X_test, y_train, y_valid, y_test = split(data, valid_size=0.2, test_size=0.2, random_state=42)
    X_train, y_train = remove_outlier(X_train, y_train)
    X_train, X_valid, X_test = imputation(X_train, X_valid, X_test)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

# Feature Importance Visualization (using LightGBM model)
def plot_feature_importance(model, feature_names, plot_dir='plots'):
    plt.figure(figsize=(10, 8))
    plot_importance(model, max_num_features=10, importance_type='split', xlabel='Importance', grid=False)
    plt.title("Feature Importance")
    filepath = os.path.join(plot_dir, 'feature_importance.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved at: {filepath}")

# SHAP Analysis (using LightGBM model)
def plot_shap_values(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary plot (bar)
    filepath_bar = os.path.join(plot_dir, 'shap_summary_bar.png')
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(filepath_bar)
    plt.close()
    print(f"SHAP summary bar plot saved at: {filepath_bar}")

    # SHAP summary plot
    filepath_summary = os.path.join(plot_dir, 'shap_summary.png')
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(filepath_summary)
    plt.close()
    print(f"SHAP summary plot saved at: {filepath_summary}")

# Tree Structure Visualization (using DecisionTreeClassifier model)
def plot_decision_tree(model, plot_dir='plots', figsize=(20, 10), dpi=300, max_depth=4, fontsize=14):
    filepath = os.path.join(plot_dir, 'decision_tree_structure.png')
    plt.figure(figsize=figsize, dpi=dpi)
    plot_tree(
        model, 
        filled=True, 
        feature_names=X_train.columns, 
        rounded=True, 
        fontsize=fontsize, 
        max_depth=max_depth 
    )
    
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Decision tree structure plot saved at: {filepath}")

if __name__ == "__main__":
    # Load models
    lgb_model = load_model("lightgbm")
    tree_model = load_model("decisiontree")
    
    # Load and preprocess data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_preprocess_data()
    
    # Create plot directory if it doesn't exist
    plot_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Perform analysis and save plots
    print("Displaying and saving feature importance (LightGBM)...")
    plot_feature_importance(lgb_model, X_test.columns)

    print("Calculating, displaying, and saving SHAP values (LightGBM)...")
    plot_shap_values(lgb_model, X_test)

    print("Displaying and saving tree structure (Decision Tree)...")
    plot_decision_tree(tree_model)
