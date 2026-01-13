import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -----------------------
# CONFIG
# -----------------------
DATA_FILES = {
    "KDD": "../data/processed/processed_kdd.csv",
    "CORES": "../data/processed/processed_cores.csv",
    "NETFLOW": "../data/processed/processed_netflow.csv",
}

RESULTS_RAW_FILE = "experiment_results_raw.csv"
RESULTS_AGG_FILE = "experiment_results_final.csv"

# RKF parameters
N_SPLITS = 5
N_REPEATS = 2
RANDOM_STATE = 42

# Threshold tuning
TUNE_THRESHOLD = True
VAL_SIZE = 0.2
THRESH_GRID = np.linspace(0.05, 0.95, 19)  # coarse but stable

# Isolation Forest
IFIT_ON_NORMAL_ONLY = False  

# Stream / S4
STREAM_MAX_INITIAL = 500_000
STREAM_MAX_BATCHES = 500
STREAM_BATCH_SIZE = 1000
STREAM_RETRAIN_INTERVAL = 5
STREAM_RETRAIN_ON_DROP = True
STREAM_DROP_THRESH = 0.05
STREAM_WINDOW_SIZE = 100_000

# -----------------------
# HELPERS
# -----------------------
def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }


def best_f1_threshold(y_true, probas, grid=THRESH_GRID):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (probas >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def get_attack_rate(y):
    # attack rate = fraction of 1s
    r = float(np.mean(y))
    return float(np.clip(r, 0.001, 0.5))


def load_data():
    datasets = {}
    print("--- Loading Datasets ---")
    for name, path in DATA_FILES.items():
        if os.path.exists(path):
            print(f"Loading {name} from {path}...")
            df = pd.read_csv(path).replace([np.inf, -np.inf], 0)
            datasets[name] = df
        else:
            print(f"WARNING: {path} not found. Run preprocessing first.")
    return datasets


# -----------------------
# MODELS
# -----------------------
def make_logreg(seed):
    # Better stability for imbalanced IDS data:
    # - scaling helps optimization
    # - class_weight helps F1/Recall
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]
    )


def make_xgb(seed):
    # Light, stable params (not heavy tuning)
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
        n_jobs=-1,
    )


def make_isoforest(seed, contamination):
    return IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=200,
        n_jobs=-1,
    )


def get_models(seed, y_for_if=None):
    # Set IF contamination from training distribution when possible
    contamination = 0.1 if y_for_if is None else get_attack_rate(y_for_if)
    return {
        "M1_LogisticRegression": make_logreg(seed),
        "M2_XGBoost": make_xgb(seed),
        "M3_IsolationForest": make_isoforest(seed, contamination),
    }


# -----------------------
# PREDICTIONS 
# -----------------------
def fit_and_predict(model, model_name, X_train, y_train, X_test, tune_threshold=TUNE_THRESHOLD):
    """
    For LogReg/XGB: optionally tune threshold on a validation split from TRAIN ONLY,
    then refit on full train and predict on test.
    For IsolationForest: fit unsupervised, map {-1,1} -> {1,0}.
    """
    if "IsolationForest" in model_name:
        if IFIT_ON_NORMAL_ONLY:
            X_fit = X_train[y_train == 0]
            model.fit(X_fit)
        else:
            model.fit(X_train)
        preds_raw = model.predict(X_test)
        preds = np.where(preds_raw == -1, 1, 0)
        return preds, None

    # Supervised models: tune threshold on train split only
    threshold = 0.5
    if tune_threshold:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
        (tr_idx, va_idx) = next(splitter.split(X_train, y_train))
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        model.fit(X_tr, y_tr)
        probas = model.predict_proba(X_va)[:, 1]
        threshold, _ = best_f1_threshold(y_va, probas)

    # Refit on full training set
    model.fit(X_train, y_train)
    test_probas = model.predict_proba(X_test)[:, 1]
    preds = (test_probas >= threshold).astype(int)
    return preds, threshold


# -----------------------
# SCENARIOS
# -----------------------
def run_s1_baseline(datasets, results):
    print("\n--- Running S1: Baseline (Repeated Stratified K-Fold) ---")
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    for name, df in datasets.items():
        X = df.drop("label", axis=1)
        y = df["label"]

        fold_idx = 0
        for train_index, test_index in rskf.split(X, y):
            fold_idx += 1
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            models = get_models(seed=RANDOM_STATE + fold_idx, y_for_if=y_train)

            for model_name, model in models.items():
                preds, thr = fit_and_predict(model, model_name, X_train, y_train, X_test)
                metrics = calculate_metrics(y_test, preds)

                row = {
                    "Scenario": "S1_Baseline",
                    "Train_Set": name,
                    "Test_Set": name,
                    "Model": model_name,
                    "Fold": fold_idx,
                    **metrics,
                }
                if thr is not None:
                    row["Threshold"] = thr
                results.append(row)

        print(f"  > {name} processed.")


def run_s2_transfer(datasets, results):
    print("\n--- Running S2: Transfer Learning (Train on A -> Test on B) ---")
    dataset_names = list(datasets.keys())

    for train_name in dataset_names:
        for test_name in dataset_names:
            if train_name == test_name:
                continue

            print(f"  > Train: {train_name} -> Test: {test_name}")
            train_df = datasets[train_name]
            test_df = datasets[test_name]

            X_train = train_df.drop("label", axis=1)
            y_train = train_df["label"]
            X_test = test_df.drop("label", axis=1)
            y_test = test_df["label"]

            for i in range(N_REPEATS):
                models = get_models(seed=RANDOM_STATE + i, y_for_if=y_train)

                for model_name, model in models.items():
                    preds, thr = fit_and_predict(model, model_name, X_train, y_train, X_test)
                    metrics = calculate_metrics(y_test, preds)

                    row = {
                        "Scenario": "S2_Transfer",
                        "Train_Set": train_name,
                        "Test_Set": test_name,
                        "Model": model_name,
                        "Fold": i + 1,
                        **metrics,
                    }
                    if thr is not None:
                        row["Threshold"] = thr
                    results.append(row)


def run_s3_combined(datasets, results):
    print("\n--- Running S3: Combined Training (Train on A+B -> Test on C) ---")
    dataset_names = list(datasets.keys())

    for test_name in dataset_names:
        train_dfs = [df for name, df in datasets.items() if name != test_name]
        if not train_dfs:
            continue

        combined_train = pd.concat(train_dfs, ignore_index=True)
        train_source_names = "+".join([name for name in dataset_names if name != test_name])

        print(f"  > Train: [{train_source_names}] -> Test: {test_name}")

        X_train = combined_train.drop("label", axis=1)
        y_train = combined_train["label"]
        X_test = datasets[test_name].drop("label", axis=1)
        y_test = datasets[test_name]["label"]

        for i in range(N_REPEATS):
            models = get_models(seed=RANDOM_STATE + i, y_for_if=y_train)

            for model_name, model in models.items():
                preds, thr = fit_and_predict(model, model_name, X_train, y_train, X_test)
                metrics = calculate_metrics(y_test, preds)

                row = {
                    "Scenario": "S3_Combined",
                    "Train_Set": train_source_names,
                    "Test_Set": test_name,
                    "Model": model_name,
                    "Fold": i + 1,
                    **metrics,
                }
                if thr is not None:
                    row["Threshold"] = thr
                results.append(row)


def run_s4_stream(datasets):
    """
    Streaming extension:
    - initial training on combined (other datasets)
    - evaluate batch by batch on stream dataset
    - retrain on sliding window when F1 drops (controlled retraining)
    - saves F1-over-time plots (by dataset, by model)
    """
    print("\n--- Running S4: Stream (Controlled Retraining + Sliding Window) ---")

    dataset_names = list(datasets.keys())
    stream_results = []

    for test_name in dataset_names:
        train_dfs = [df for name, df in datasets.items() if name != test_name]
        if not train_dfs:
            continue

        combined_train = pd.concat(train_dfs, ignore_index=True)
        if len(combined_train) > STREAM_MAX_INITIAL:
            combined_train = combined_train.sample(n=STREAM_MAX_INITIAL, random_state=RANDOM_STATE).reset_index(drop=True)

        X_train_initial = combined_train.drop("label", axis=1)
        y_train_initial = combined_train["label"]

        target_df = datasets[test_name]
        X_stream = target_df.drop("label", axis=1)
        y_stream = target_df["label"]

        train_source_names = "+".join([name for name in dataset_names if name != test_name])
        print(f"  > Initial Base: [{train_source_names}] (Size: {len(X_train_initial)}) -> Stream: {test_name}")

        models = get_models(seed=RANDOM_STATE, y_for_if=y_train_initial)
        training_sets = {}
        last_f1 = {}
        thresholds = {}

        # Initial training + initial threshold selection (supervised only)
        for m_name, model in models.items():
            training_sets[m_name] = {
                "X": X_train_initial.copy().reset_index(drop=True),
                "y": y_train_initial.copy().reset_index(drop=True),
            }
            last_f1[m_name] = None

            print(f"    Training initial {m_name}...", end=" ")
            if "IsolationForest" in m_name:
                if IFIT_ON_NORMAL_ONLY:
                    model.fit(X_train_initial[y_train_initial == 0])
                else:
                    model.fit(X_train_initial)
                thresholds[m_name] = None
            else:
                # tune threshold on training only
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
                tr_idx, va_idx = next(splitter.split(X_train_initial, y_train_initial))
                X_tr, y_tr = X_train_initial.iloc[tr_idx], y_train_initial.iloc[tr_idx]
                X_va, y_va = X_train_initial.iloc[va_idx], y_train_initial.iloc[va_idx]
                model.fit(X_tr, y_tr)
                thr, _ = best_f1_threshold(y_va, model.predict_proba(X_va)[:, 1])
                thresholds[m_name] = thr
                model.fit(X_train_initial, y_train_initial)
            print("Done.")

        n_batches = int(np.ceil(len(X_stream) / STREAM_BATCH_SIZE))
        n_batches = min(n_batches, STREAM_MAX_BATCHES)

        for i in range(n_batches):
            start_idx = i * STREAM_BATCH_SIZE
            end_idx = min((i + 1) * STREAM_BATCH_SIZE, len(X_stream))

            X_batch = X_stream.iloc[start_idx:end_idx].reset_index(drop=True)
            y_batch = y_stream.iloc[start_idx:end_idx].reset_index(drop=True)

            for model_name, model in models.items():
                if "IsolationForest" in model_name:
                    preds_raw = model.predict(X_batch)
                    preds = np.where(preds_raw == -1, 1, 0)
                else:
                    probas = model.predict_proba(X_batch)[:, 1]
                    preds = (probas >= (thresholds[model_name] if thresholds[model_name] is not None else 0.5)).astype(int)

                f1 = f1_score(y_batch, preds, zero_division=0)
                stream_results.append({
                    "Scenario": "S4_Stream",
                    "Test_Stream": test_name,
                    "Model": model_name,
                    "Batch_Idx": i,
                    "F1-Score": f1,
                })

                # Controlled retraining with sliding window
                do_retrain = False
                if (i % STREAM_RETRAIN_INTERVAL == 0) and STREAM_RETRAIN_ON_DROP:
                    if last_f1[model_name] is not None and (last_f1[model_name] - f1) >= STREAM_DROP_THRESH:
                        do_retrain = True

                if do_retrain:
                    old_X = training_sets[model_name]["X"]
                    old_y = training_sets[model_name]["y"]
                    current_X = pd.concat([old_X, X_batch], ignore_index=True)
                    current_y = pd.concat([old_y, y_batch], ignore_index=True)

                    if len(current_X) > STREAM_WINDOW_SIZE:
                        current_X = current_X.iloc[-STREAM_WINDOW_SIZE:].reset_index(drop=True)
                        current_y = current_y.iloc[-STREAM_WINDOW_SIZE:].reset_index(drop=True)

                    training_sets[model_name]["X"] = current_X
                    training_sets[model_name]["y"] = current_y

                    print(f"\n    [Retrain] {model_name} | batch={i} | train_size={len(current_X)}", end=" ")

                    # update contamination to new window distribution
                    if "IsolationForest" in model_name:
                        models[model_name] = make_isoforest(RANDOM_STATE, contamination=get_attack_rate(current_y))
                        model = models[model_name]
                        if IFIT_ON_NORMAL_ONLY:
                            model.fit(current_X[current_y == 0])
                        else:
                            model.fit(current_X)
                        thresholds[model_name] = None
                    else:
                        # retune threshold on window (train-only)
                        splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
                        tr_idx, va_idx = next(splitter.split(current_X, current_y))
                        X_tr, y_tr = current_X.iloc[tr_idx], current_y.iloc[tr_idx]
                        X_va, y_va = current_X.iloc[va_idx], current_y.iloc[va_idx]
                        model.fit(X_tr, y_tr)
                        thr, _ = best_f1_threshold(y_va, model.predict_proba(X_va)[:, 1])
                        thresholds[model_name] = thr
                        model.fit(current_X, current_y)

                    print("Done.")

                last_f1[model_name] = f1

            if i % 10 == 0:
                print(".", end="", flush=True)

        print(" Stream Done!")

    if stream_results:
        df_stream = pd.DataFrame(stream_results)
        df_stream.to_csv("experiment_stream_results.csv", index=False)

        ROLLING_WINDOW = 10
        df_stream['F1_Smoothed'] = (
        df_stream
        .groupby(['Test_Stream', 'Model'])['F1-Score']
        .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
        # Plot: by dataset stream
        for stream in df_stream["Test_Stream"].unique():
            plt.figure(figsize=(10, 6))
            sub = df_stream[df_stream["Test_Stream"] == stream]
            for m in sub["Model"].unique():
                subm = sub[sub["Model"] == m]
                plt.plot(subm["Batch_Idx"], subm["F1_Smoothed"], label=m)
            plt.title(f"S4 Stream: {stream} (F1 over time)")
            plt.xlabel("Batch Index")
            plt.ylabel("F1 Score")
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            fname = f"plot_stream_by_DATA_{stream}.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"Saved {fname}")

        # Plot: by model
        for model in df_stream["Model"].unique():
            plt.figure(figsize=(10, 6))
            sub = df_stream[df_stream["Model"] == model]
            for s in sub["Test_Stream"].unique():
                subs = sub[sub["Test_Stream"] == s]
                plt.plot(subm["Batch_Idx"], subm["F1_Smoothed"], label=s)
            plt.title(f"S4 Stream: {model} (F1 over time)")
            plt.xlabel("Batch Index")
            plt.ylabel("F1 Score")
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            fname = f"plot_stream_by_MODEL_{model}.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"Saved {fname}")


# -----------------------
# MAIN
# -----------------------
def main():
    datasets = load_data()
    if not datasets:
        print("No datasets loaded. Exiting.")
        return

    results = []

    run_s1_baseline(datasets, results)
    run_s2_transfer(datasets, results)
    run_s3_combined(datasets, results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_RAW_FILE, index=False)

    summary = results_df.groupby(["Scenario", "Train_Set", "Test_Set", "Model"])["F1-Score"].agg(["mean", "std"]).reset_index()
    summary.to_csv(RESULTS_AGG_FILE, index=False)

    print(f"\nSaved raw results to {RESULTS_RAW_FILE}")
    print(f"Saved aggregated results to {RESULTS_AGG_FILE}")
    print(summary.head())

    run_s4_stream(datasets)


if __name__ == "__main__":
    main()
