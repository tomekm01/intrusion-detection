import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
from xgboost import XGBClassifier
import os
import warnings

warnings.filterwarnings('ignore')

DATA_FILES = {
    'KDD': '../data/processed/processed_kdd.csv',
    'CORES': '../data/processed/processed_cores.csv',
    'NETFLOW': '../data/processed/processed_netflow.csv'
}

RESULTS_FILE = 'experiment_results_final.csv'
STREAM_PLOT_FILE = 'stream_learning_process.png'

# RKF parameters
N_SPLITS = 5
N_REPEATS = 2
RANDOM_STATE = 42


def get_models(seed=RANDOM_STATE):
    return {
        'M1_LogisticRegression': LogisticRegression(max_iter=1000, random_state=seed),
        'M2_XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=seed, verbosity=0),
        'M3_IsolationForest': IsolationForest(contamination=0.1, random_state=seed)
    }


def calculate_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }


def get_predictions(model, X, model_name):
    if 'IsolationForest' in model_name:
        preds_raw = model.predict(X)
        return np.where(preds_raw == -1, 1, 0)
    else:
        return model.predict(X)


def load_data():
    datasets = {}
    print("--- Loading Datasets ---")
    for name, path in DATA_FILES.items():
        if os.path.exists(path):
            print(f"Loading {name} from {path}...")
            df = pd.read_csv(path)
            df = df.replace([np.inf, -np.inf], 0)
            datasets[name] = df
        else:
            print(f"WARNING: {path} not found. Ensure you ran the processing script first.")
    return datasets

# --- S1: BASELINE (Repeated K-Fold) ---
def run_s1_baseline(datasets, results):
    print("\n--- Running S1: Baseline (Repeated K-Fold) ---")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    for name, df in datasets.items():
        X = df.drop('label', axis=1)
        y = df['label']

        fold_idx = 0
        for train_index, test_index in rskf.split(X, y):
            fold_idx += 1
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            models = get_models(seed=RANDOM_STATE + fold_idx)
            for model_name, model in models.items():
                if 'IsolationForest' in model_name:
                    model.fit(X_train)
                else:
                    model.fit(X_train, y_train)

                preds = get_predictions(model, X_test, model_name)
                metrics = calculate_metrics(y_test, preds)

                results.append({
                    'Scenario': 'S1_Baseline',
                    'Train_Set': name, 'Test_Set': name,
                    'Model': model_name, 'Fold': fold_idx,
                    **metrics
                })
        print(f"  > {name} processed.")


# --- S2: TRANSFER LEARNING ---
def run_s2_transfer(datasets, results):
    print("\n--- Running S2: Transfer Learning ---")
    dataset_names = list(datasets.keys())

    for train_name in dataset_names:
        for test_name in dataset_names:
            if train_name == test_name: continue

            print(f"  > Train: {train_name} -> Test: {test_name}")
            X_train = datasets[train_name].drop('label', axis=1)
            y_train = datasets[train_name]['label']
            X_test = datasets[test_name].drop('label', axis=1)
            y_test = datasets[test_name]['label']

            for i in range(N_REPEATS):
                models = get_models(seed=RANDOM_STATE + i)
                for model_name, model in models.items():
                    if 'IsolationForest' in model_name:
                        model.fit(X_train)
                    else:
                        model.fit(X_train, y_train)

                    preds = get_predictions(model, X_test, model_name)
                    metrics = calculate_metrics(y_test, preds)
                    results.append({
                        'Scenario': 'S2_Transfer',
                        'Train_Set': train_name, 'Test_Set': test_name,
                        'Model': model_name, 'Fold': i + 1,
                        **metrics
                    })


# --- S3: COMBINED TRAINING ---
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
        X_test = datasets[test_name].drop('label', axis=1)
        y_test = datasets[test_name]['label']
        for i in range(N_REPEATS):
            models = get_models(seed=RANDOM_STATE + i)
            for model_name, model in models.items():
                if 'IsolationForest' in model_name:
                    model.fit(X_train)
                else:
                    model.fit(X_train, y_train)
                preds = get_predictions(model, X_test, model_name)
                metrics = calculate_metrics(y_test, preds)
                results.append({
                    'Scenario': 'S3_Combined',
                    'Train_Set': train_source_names, 'Test_Set': test_name,
                    'Model': model_name, 'Fold': i + 1,
                    **metrics
                })


# --- S4: STREAM WITH RETRAINING (Hybrid Solution) ---
def run_s4_stream(datasets):
    MAX_INITIAL = 500_000  # losowa próbka do initial training
    MAX_BATCHES = 500  # maksymalnie batchy
    RETRAIN_ON_EACH_BATCH = True
    RETRAIN_INTERVAL = 5  # interwał sprawdzania
    RETRAIN_ON_DROP_THRESH = 0.05  # próg spadku
    BATCH_SIZE_S4 = 1000
    WINDOW_SIZE = 100_000  # wielkość okna przesuwnego (FIFO)

    print(f"\n--- Running S4: Stream (Controlled Retraining + FIFO Window) ---")
    dataset_names = list(datasets.keys())
    stream_results = []

    for test_name in dataset_names:
        train_dfs = [df for name, df in datasets.items() if name != test_name]
        if not train_dfs:
            continue

        combined_train = pd.concat(train_dfs, ignore_index=True)

        # === OGRANICZENIE INITIAL TRAINING ===
        if len(combined_train) > MAX_INITIAL:
            combined_train = combined_train.sample(n=MAX_INITIAL, random_state=42).reset_index(drop=True)

        train_source_names = "+".join([name for name in dataset_names if name != test_name])

        X_train_initial = combined_train.drop('label', axis=1)
        y_train_initial = combined_train['label']

        target_df = datasets[test_name]
        X_stream = target_df.drop('label', axis=1)
        y_stream = target_df['label']

        print(f"  > Initial Base: [{train_source_names}] (Size: {len(X_train_initial)}) -> Stream: {test_name}")

        models = get_models(seed=42)
        training_sets = {}
        last_f1 = {}

        for m_name, model in models.items():
            training_sets[m_name] = {
                'X': X_train_initial.copy().reset_index(drop=True),
                'y': y_train_initial.copy().reset_index(drop=True)
            }
            last_f1[m_name] = None

            print(f"    Training initial {m_name}...", end=" ")
            if 'IsolationForest' in m_name:
                model.fit(X_train_initial)
            else:
                model.fit(X_train_initial, y_train_initial)
            print("Done.")

        n_batches = int(np.ceil(len(X_stream) / BATCH_SIZE_S4))
        n_batches = min(n_batches, MAX_BATCHES)

        for i in range(n_batches):
            start_idx = i * BATCH_SIZE_S4
            end_idx = min((i + 1) * BATCH_SIZE_S4, len(X_stream))
            X_batch = X_stream.iloc[start_idx:end_idx].reset_index(drop=True)
            y_batch = y_stream.iloc[start_idx:end_idx].reset_index(drop=True)

            for model_name, model in models.items():
                preds = get_predictions(model, X_batch, model_name)
                f1 = f1_score(y_batch, preds, zero_division=0)

                stream_results.append({
                    'Test_Stream': test_name,
                    'Model': model_name,
                    'Batch_Idx': i,
                    'F1-Score': f1
                })

                # === LOGIKA RETRAININGU ===
                do_retrain = False
                if RETRAIN_ON_EACH_BATCH and (i % RETRAIN_INTERVAL == 0):
                    if last_f1[model_name] is not None:
                        if (last_f1[model_name] - f1) >= RETRAIN_ON_DROP_THRESH:
                            do_retrain = True

                if do_retrain:
                    old_X = training_sets[model_name]['X']
                    old_y = training_sets[model_name]['y']

                    # === IMPLEMENTACJA FIFO (SLIDING WINDOW) ===
                    current_X = pd.concat([old_X, X_batch], ignore_index=True)
                    current_y = pd.concat([old_y, y_batch], ignore_index=True)

                    if len(current_X) > WINDOW_SIZE:
                        # Ucinamy najstarsze (z początku), zostawiamy najnowsze (na końcu)
                        current_X = current_X.iloc[-WINDOW_SIZE:].reset_index(drop=True)
                        current_y = current_y.iloc[-WINDOW_SIZE:].reset_index(drop=True)

                    training_sets[model_name]['X'] = current_X
                    training_sets[model_name]['y'] = current_y

                    print(f"\n    [Retrain] {model_name} | batch={i} | train_size={len(current_X)}", end=" ")

                    if 'IsolationForest' in model_name:
                        model.fit(current_X)
                    else:
                        model.fit(current_X, current_y)

                    print("Done.")

                last_f1[model_name] = f1

            if i % 5 == 0:
                print(".", end="", flush=True)

        print(" Stream Done!")

    if stream_results:
        df_stream = pd.DataFrame(stream_results)

        # === NOWY KOD: RYSOWANIE WYKRESÓW ===
        print("\n Generating Datastreams plots")
        unique_streams = df_stream['Test_Stream'].unique()
        for stream in unique_streams:
            plt.figure(figsize=(10, 6))
            subset = df_stream[df_stream['Test_Stream'] == stream]
            sns.lineplot(data=subset, x='Batch_Idx', y='F1-Score', hue='Model', marker='o')
            plt.title(f'Data: {stream}')
            plt.ylabel('F1 Score')
            plt.xlabel('Batch Index')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.05)
            filename = f'plot_stream_by_DATA_{stream}.png'
            plt.savefig(filename)
            plt.close()
            print(f" Saved {filename}")

        print("\n Generating Model Plots")
        unique_models = df_stream['Model'].unique()
        for model in unique_models:
            plt.figure(figsize=(10, 6))
            subset = df_stream[df_stream['Model'] == model]
            sns.lineplot(data=subset, x='Batch_Idx', y='F1-Score', hue='Test_Stream', marker='s')
            plt.title(f'Model: {model}')
            plt.ylabel('F1 Score')
            plt.xlabel('Batch Index')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.05)
            filename = f'plot_stream_by_MODEL_{model}.png'
            plt.savefig(filename)
            plt.close()
            print(f" Saved {filename}")

        # === NOWY KOD: STATYSTYKI KOŃCOWE ===
        print("\n" + "=" * 80)
        print(" S4 STREAM STATISTICS SUMMARY")
        print("=" * 80)

        for stream in unique_streams:
            print(f"\nDataset Stream: {stream}")
            # Nagłówek tabeli
            print(f"{'Model':<25} | {'Mean F1':<8} | {'Max F1':<8} (Batch) | {'Min F1':<8} (Batch)")
            print("-" * 75)

            subset_stream = df_stream[df_stream['Test_Stream'] == stream]

            for model in subset_stream['Model'].unique():
                subset_model = subset_stream[subset_stream['Model'] == model]

                # Obliczenia
                mean_val = subset_model['F1-Score'].mean()

                # Znalezienie indeksu (wiersza) z max wartością
                idx_max = subset_model['F1-Score'].idxmax()
                max_val = subset_model.loc[idx_max, 'F1-Score']
                batch_max = subset_model.loc[idx_max, 'Batch_Idx']

                # Znalezienie indeksu (wiersza) z min wartością
                idx_min = subset_model['F1-Score'].idxmin()
                min_val = subset_model.loc[idx_min, 'F1-Score']
                batch_min = subset_model.loc[idx_min, 'Batch_Idx']

                # Wypisanie
                print(
                    f"{model:<25} | {mean_val:.4f}   | {max_val:.4f}   ({batch_max:>3}) | {min_val:.4f}   ({batch_min:>3})")

        print("=" * 80 + "\n")

def main():
    datasets = load_data()
    if not datasets:
        print("No datasets loaded. Exiting.")
        return

    results = []

    # Scenarios 1-3
    run_s1_baseline(datasets, results)
    run_s2_transfer(datasets, results)
    run_s3_combined(datasets, results)

    # Save S1-S3 Results
    results_df = pd.DataFrame(results)
    # Saving raw data
    results_df.to_csv('experiment_results_raw.csv', index=False)

    # Agregated results (Mean/Std)
    summary = results_df.groupby(['Scenario', 'Train_Set', 'Test_Set', 'Model'])['F1-Score'].agg(
        ['mean', 'std']).reset_index()
    summary.to_csv(RESULTS_FILE, index=False)
    print(f"\nAggregated results saved to {RESULTS_FILE}")
    print(summary.head())

    # Scenario 4 (Stream)
    run_s4_stream(datasets)


if __name__ == "__main__":
    main()

#WYRESY SĄ DLA s4 z podanymi warunkami: RETRAIN_ON_EACH_BATCH = True RETRAIN_INTERVAL = 5 BATCH_SIZE_S4 = 1000 WINDOW_SIZE = 500000 #none or huge number like 500 000