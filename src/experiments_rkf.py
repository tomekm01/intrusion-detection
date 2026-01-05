import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import os

# --- KONFIGURACJA WALIDACJI ---
N_SPLITS = 5
N_REPEATS = 2
RANDOM_STATE = 42

DATA_FILES = {
    'KDD': '../data/processed/processed_kdd.csv',
    'CORES': '../data/processed/processed_cores.csv',
    'NETFLOW': '../data/processed/processed_netflow.csv'
}

RESULTS_FILE = 'experiment_results_rkf.csv'


def get_models():
    return {
        'M1_LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'M2_XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
        'M3_IsolationForest': IsolationForest(contamination=0.1, random_state=RANDOM_STATE)
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
            # Obsługa nieskończoności
            df = df.replace([np.inf, -np.inf], 0)
            datasets[name] = df
        else:
            print(f"WARNING: {path} not found. Skipping.")
    return datasets


def run_s1_baseline(datasets, results):
    print("\n--- Running S1: Baseline (RepeatedKFold CV) ---")
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    for name, df in datasets.items():
        X = df.drop('label', axis=1)
        y = df['label']

        # Iteracja przez foldy
        for i, (train_index, test_index) in enumerate(rkf.split(X, y)):
            # Używamy .iloc dla indeksów z sklearn
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            fold_id = f"Rep_{(i // N_SPLITS) + 1}_Fold_{(i % N_SPLITS) + 1}"

            models = get_models()
            for model_name, model in models.items():
                # print(f"  > {name} | {fold_id} | {model_name}") # Odkomentuj dla pełnego logu

                model.fit(X_train, y_train) if 'IsolationForest' not in model_name else model.fit(X_train)
                metrics = evaluate_model(model, X_test, y_test, model_name)

                results.append({
                    'Scenario': 'S1_Baseline',
                    'Train_Set': name,
                    'Test_Set': name,
                    'Model': model_name,
                    'Fold_Info': fold_id,
                    **metrics
                })
        print(f"  > {name} processing complete.")


def run_s2_transfer(datasets, results):
    print("\n--- Running S2: Transfer Learning (Train CV -> Fixed External Test) ---")
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    dataset_names = list(datasets.keys())
    for train_name in dataset_names:
        for test_name in dataset_names:
            if train_name == test_name: continue

            print(f"  > Train Source: {train_name} -> Target: {test_name}")

            # Dane źródłowe (do trenowania)
            train_df = datasets[train_name]
            X_source = train_df.drop('label', axis=1)
            y_source = train_df['label']

            # Dane docelowe (fixed test set)
            test_df = datasets[test_name]
            X_target_fixed = test_df.drop('label', axis=1)
            y_target_fixed = test_df['label']

            # RepeatedKFold na danych ŹRÓDŁOWYCH
            # Trenujemy na podzbiorze źródła, testujemy na CAŁYM celu
            for i, (train_index, _) in enumerate(rkf.split(X_source, y_source)):
                X_train_fold = X_source.iloc[train_index]
                y_train_fold = y_source.iloc[train_index]

                fold_id = f"Rep_{(i // N_SPLITS) + 1}_Fold_{(i % N_SPLITS) + 1}"

                models = get_models()
                for model_name, model in models.items():
                    model.fit(X_train_fold, y_train_fold) if 'IsolationForest' not in model_name else model.fit(
                        X_train_fold)

                    # Testujemy zawsze na zewnętrznym zbiorze (Target)
                    metrics = evaluate_model(model, X_target_fixed, y_target_fixed, model_name)

                    results.append({
                        'Scenario': 'S2_Transfer',
                        'Train_Set': train_name,
                        'Test_Set': test_name,
                        'Model': model_name,
                        'Fold_Info': fold_id,
                        **metrics
                    })


def run_s3_combined(datasets, results):
    print("\n--- Running S3: Combined Training (Combined CV -> Fixed External Test) ---")
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    dataset_names = list(datasets.keys())
    for test_name in dataset_names:
        # Przygotowanie zbioru treningowego (połączone wszystko oprócz testowego)
        train_dfs = [df for name, df in datasets.items() if name != test_name]
        if not train_dfs: continue

        combined_train = pd.concat(train_dfs, ignore_index=True)
        train_source_names = "+".join([name for name in dataset_names if name != test_name])

        X_combined = combined_train.drop('label', axis=1)
        y_combined = combined_train['label']

        # Zbiór testowy (fixed)
        test_df = datasets[test_name]
        X_target_fixed = test_df.drop('label', axis=1)
        y_target_fixed = test_df['label']

        print(f"  > Train: [{train_source_names}] -> Test: {test_name}")

        # RepeatedKFold na połączonym zbiorze treningowym
        for i, (train_index, _) in enumerate(rkf.split(X_combined, y_combined)):
            X_train_fold = X_combined.iloc[train_index]
            y_train_fold = y_combined.iloc[train_index]

            fold_id = f"Rep_{(i // N_SPLITS) + 1}_Fold_{(i % N_SPLITS) + 1}"

            models = get_models()
            for model_name, model in models.items():
                model.fit(X_train_fold, y_train_fold) if 'IsolationForest' not in model_name else model.fit(
                    X_train_fold)

                metrics = evaluate_model(model, X_target_fixed, y_target_fixed, model_name)

                results.append({
                    'Scenario': 'S3_Combined',
                    'Train_Set': train_source_names,
                    'Test_Set': test_name,
                    'Model': model_name,
                    'Fold_Info': fold_id,
                    **metrics
                })


def main():
    datasets = load_data()
    if not datasets:
        print("No datasets loaded. Check file paths.")
        return
    results = []

    # Uruchamianie scenariuszy
    run_s1_baseline(datasets, results)
    run_s2_transfer(datasets, results)
    run_s3_combined(datasets, results)

    # Zapis wyników
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)

    print(f"\nExperiment Complete! Results saved to {RESULTS_FILE}")

    # Wyświetlenie uśrednionych wyników (grupowanie po foldach)
    summary = results_df.groupby(['Scenario', 'Train_Set', 'Test_Set', 'Model'])['F1-Score'].agg(
        ['mean', 'std']).reset_index()
    print("\nSummary of F1-Scores (Mean over folds):")
    print(summary)


if __name__ == "__main__":
    main()