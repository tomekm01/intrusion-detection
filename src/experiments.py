import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import os

DATA_FILES = {
    'KDD': '../data/processed/processed_kdd.csv',
    'CORES': '../data/processed/processed_cores.csv',
    'NETFLOW': '../data/processed/processed_netflow.csv'
}

RESULTS_FILE = 'experiment_results.csv'

def get_models():
    return {
        'M1_LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'M2_XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'M3_IsolationForest': IsolationForest(contamination=0.1, random_state=42)
    }


def evaluate_model(model, X_test, y_test, model_name):
    if 'IsolationForest' in model_name:
        preds_raw = model.predict(X_test)
        preds = np.where(preds_raw == -1, 1, 0)
    else:
        preds = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, zero_division=0),
        'Recall': recall_score(y_test, preds, zero_division=0),
        'F1-Score': f1_score(y_test, preds, zero_division=0)
    }


def load_data():
    datasets = {}
    for name, path in DATA_FILES.items():
        if os.path.exists(path):
            print(f"Loading {name}...")
            df = pd.read_csv(path)
            df = df.replace([np.inf, -np.inf], 0)
            datasets[name] = df
        else:
            print(f"WARNING: {path} not found. Skipping.")
    return datasets


def run_s1_baseline(datasets, results):
    print("\n--- Running S1: Baseline (Train/Test on same dataset) ---")
    for name, df in datasets.items():
        X = df.drop('label', axis=1)
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = get_models()
        for model_name, model in models.items():
            print(f"  > {name} | {model_name}")
            model.fit(X_train, y_train) if 'IsolationForest' not in model_name else model.fit(X_train)
            metrics = evaluate_model(model, X_test, y_test, model_name)
            results.append({
                'Scenario': 'S1_Baseline',
                'Train_Set': name,
                'Test_Set': name,
                'Model': model_name,
                **metrics
            })


def run_s2_transfer(datasets, results):
    print("\n--- Running S2: Transfer Learning (Train on A, Test on B) ---")
    dataset_names = list(datasets.keys())
    for train_name in dataset_names:
        for test_name in dataset_names:
            if train_name == test_name: continue
            print(f"  > Train: {train_name} -> Test: {test_name}")
            train_df = datasets[train_name]
            X_train = train_df.drop('label', axis=1)
            y_train = train_df['label']
            test_df = datasets[test_name]
            X_test = test_df.drop('label', axis=1)
            y_test = test_df['label']
            models = get_models()
            for model_name, model in models.items():
                model.fit(X_train, y_train) if 'IsolationForest' not in model_name else model.fit(X_train)
                metrics = evaluate_model(model, X_test, y_test, model_name)
                results.append({
                    'Scenario': 'S2_Transfer',
                    'Train_Set': train_name,
                    'Test_Set': test_name,
                    'Model': model_name,
                    **metrics
                })

def run_s3_combined(datasets, results):
    print("\n--- Running S3: Combined Training ---")
    dataset_names = list(datasets.keys())
    for test_name in dataset_names:
        train_dfs = [df for name, df in datasets.items() if name != test_name]
        if not train_dfs: continue
        combined_train = pd.concat(train_dfs, ignore_index=True)
        train_source_names = "+".join([name for name in dataset_names if name != test_name])
        print(f"  > Train: [{train_source_names}] -> Test: {test_name}")
        X_train = combined_train.drop('label', axis=1)
        y_train = combined_train['label']
        test_df = datasets[test_name]
        X_test = test_df.drop('label', axis=1)
        y_test = test_df['label']
        models = get_models()
        for model_name, model in models.items():
            model.fit(X_train, y_train) if 'IsolationForest' not in model_name else model.fit(X_train)
            metrics = evaluate_model(model, X_test, y_test, model_name)
            results.append({
                'Scenario': 'S3_Combined',
                'Train_Set': train_source_names,
                'Test_Set': test_name,
                'Model': model_name,
                **metrics
            })


def main():
    datasets = load_data()
    if not datasets:
        print("No datasets loaded. Check file paths.")
        return
    results = []

    run_s1_baseline(datasets, results)
    run_s2_transfer(datasets, results)
    run_s3_combined(datasets, results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)
    
    print(f"\nExperiment Complete! Results saved to {RESULTS_FILE}")
    print("\nSummary of F1-Scores:")
    print(results_df[['Scenario', 'Train_Set', 'Test_Set', 'Model', 'F1-Score']])

if __name__ == "__main__":
    main()