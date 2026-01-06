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

# --- KONFIGURACJA ---
warnings.filterwarnings('ignore')

# Ścieżki do plików (zgodne z Twoim skryptem przetwarzającym)
DATA_FILES = {
    'KDD': '../data/processed/processed_kdd.csv',
    'CORES': '../data/processed/processed_cores.csv',
    'NETFLOW': '../data/processed/processed_netflow.csv'
}

RESULTS_FILE = 'experiment_results_final.csv'
STREAM_PLOT_FILE = 'stream_learning_process.png'

# Parametry walidacji
N_SPLITS = 5
N_REPEATS = 2
RANDOM_STATE = 42

# Parametry Strumienia (S4)
STREAM_BATCH_SIZE = 500  # Rozmiar paczki danych w strumieniu
RETRAIN_ON_EACH_BATCH = True  # TRUE = model doucza się na każdej nowej paczce (logika z Kodu 1)


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
            # Zabezpieczenie na wypadek nieskończoności z logarytmów
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
# --- ZMODYFIKOWANE S4: SZYBSZE DZIAŁANIE ---
def run_s4_stream(datasets):
    # --- USTAWIENIA (Szybkość i Wygląd) ---
    RETRAIN_INTERVAL = 10  # Douczanie co 10-tą paczkę
    BATCH_SIZE_S4 = 2000  # Rozmiar paczki
    WINDOW_SIZE = 50000  # Ile ostatnich próbek pamiętać (żeby RAM nie wybuchł)

    print(f"\n--- Running S4: Stream Simulation (Retrain Interval={RETRAIN_INTERVAL}) ---")

    dataset_names = list(datasets.keys())
    stream_results = []

    # === CZĘŚĆ OBLICZENIOWA (Ta sama co wcześniej, zoptymalizowana) ===
    for test_name in dataset_names:
        train_dfs = [df for name, df in datasets.items() if name != test_name]
        if not train_dfs: continue

        combined_train = pd.concat(train_dfs, ignore_index=True)
        # Optymalizacja startowa
        if len(combined_train) > 50000:
            combined_train = combined_train.sample(50000, random_state=42)

        X_train_initial = combined_train.drop('label', axis=1)
        y_train_initial = combined_train['label']

        target_df = datasets[test_name]
        X_stream = target_df.drop('label', axis=1)
        y_stream = target_df['label']

        print(f"  > Streaming on: {test_name} (Size: {len(X_stream)})")

        models = get_models(seed=42)
        training_sets = {}

        # Initial fit
        for m_name in models:
            training_sets[m_name] = {'X': X_train_initial.copy(), 'y': y_train_initial.copy()}
            if 'IsolationForest' in m_name:
                models[m_name].fit(X_train_initial)
            else:
                models[m_name].fit(X_train_initial, y_train_initial)

        n_batches = int(np.ceil(len(X_stream) / BATCH_SIZE_S4))

        # Pętla po paczkach
        for i in range(n_batches):
            start_idx = i * BATCH_SIZE_S4
            end_idx = min((i + 1) * BATCH_SIZE_S4, len(X_stream))

            X_batch = X_stream.iloc[start_idx:end_idx]
            y_batch = y_stream.iloc[start_idx:end_idx]

            if len(y_batch.unique()) < 2: pass

            for model_name, model in models.items():
                # 1. Ewaluacja
                preds = get_predictions(model, X_batch, model_name)
                f1 = f1_score(y_batch, preds, zero_division=0)
                stream_results.append({
                    'Test_Stream': test_name,
                    'Model': model_name,
                    'Batch_Idx': i,
                    'F1-Score': f1
                })

                # 2. Douczanie (co 10 paczek)
                if RETRAIN_ON_EACH_BATCH and (i % RETRAIN_INTERVAL == 0):
                    current_X = pd.concat([training_sets[model_name]['X'], X_batch], ignore_index=True)
                    current_y = pd.concat([training_sets[model_name]['y'], y_batch], ignore_index=True)

                    # Limit pamięci (Windowing)
                    if len(current_X) > WINDOW_SIZE:
                        current_X = current_X.iloc[-WINDOW_SIZE:]
                        current_y = current_y.iloc[-WINDOW_SIZE:]

                    training_sets[model_name]['X'] = current_X
                    training_sets[model_name]['y'] = current_y

                    if 'IsolationForest' in model_name:
                        model.fit(current_X)
                    else:
                        model.fit(current_X, current_y)

            # Kropka co 5 batchy, żeby nie śmiecić w konsoli
            if i % 5 == 0: print(".", end="", flush=True)
        print(" Done!")

    # === CZĘŚĆ WIZUALIZACYJNA (Nowa logika) ===
    if stream_results:
        df_stream = pd.DataFrame(stream_results)

        # --- TYP 1: Jeden plik na każdy ZBIÓR DANYCH (Porównanie Modeli) ---
        print("\nGenerowanie wykresów wg ZBIORÓW DANYCH...")
        unique_streams = df_stream['Test_Stream'].unique()

        for stream in unique_streams:
            plt.figure(figsize=(10, 6))
            subset = df_stream[df_stream['Test_Stream'] == stream]

            sns.lineplot(data=subset, x='Batch_Idx', y='F1-Score', hue='Model', marker='o')

            plt.title(f'Wyniki dla zbioru: {stream}')
            plt.ylabel('F1 Score')
            plt.xlabel('Upływ czasu (Batch Index)')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.05)  # Sztywna skala Y, żeby łatwiej porównywać

            filename = f'plot_stream_by_DATA_{stream}.png'
            plt.savefig(filename)
            plt.close()  # Ważne: zamykamy wykres, żeby nie nakładał się na następny
            print(f"  > Zapisano: {filename}")

        # --- TYP 2: Jeden plik na każdy MODEL (Porównanie Zbiorów) ---
        print("\nGenerowanie wykresów wg MODELI...")
        unique_models = df_stream['Model'].unique()

        for model in unique_models:
            plt.figure(figsize=(10, 6))
            subset = df_stream[df_stream['Model'] == model]

            # Tu 'hue' to Test_Stream, bo chcemy widzieć jak model radzi sobie na różnych danych
            sns.lineplot(data=subset, x='Batch_Idx', y='F1-Score', hue='Test_Stream', marker='s')

            plt.title(f'Stabilność modelu: {model}')
            plt.ylabel('F1 Score')
            plt.xlabel('Upływ czasu (Batch Index)')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.05)

            filename = f'plot_stream_by_MODEL_{model}.png'
            plt.savefig(filename)
            plt.close()
            print(f"  > Zapisano: {filename}")


def main():
    datasets = load_data()
    if not datasets:
        print("No datasets loaded. Exiting.")
        return

    results = []

    # Scenarios 1-3
    #run_s1_baseline(datasets, results)
    #run_s2_transfer(datasets, results)
    #run_s3_combined(datasets, results)

    # Save S1-S3 Results
    #results_df = pd.DataFrame(results)
    # Zapisz pełne wyniki (surowe)
    #results_df.to_csv('experiment_results_raw.csv', index=False)

    # Zapisz wyniki zagregowane (Mean/Std)
    #summary = results_df.groupby(['Scenario', 'Train_Set', 'Test_Set', 'Model'])['F1-Score'].agg(
        #['mean', 'std']).reset_index()
    #summary.to_csv(RESULTS_FILE, index=False)
    #print(f"\nAggregated results saved to {RESULTS_FILE}")
    #print(summary.head())

    # Scenario 4 (Stream)
    run_s4_stream(datasets)


if __name__ == "__main__":
    main()