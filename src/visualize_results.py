import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Konfiguracja stylu wykresów
sns.set_theme(style="whitegrid")
RESULTS_FILE = 'experiment_results_rkf.csv'


def load_results():
    try:
        df = pd.read_csv(RESULTS_FILE)
        return df
    except FileNotFoundError:
        print(f"Nie znaleziono pliku {RESULTS_FILE}. Uruchom najpierw skrypt eksperymentu.")
        return None


def plot_s1_baseline(df):
    """
    Wykres dla S1: Porównanie modeli na ich rodzimych zbiorach.
    """
    s1_data = df[df['Scenario'] == 'S1_Baseline']

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Test_Set', y='F1-Score', hue='Model', data=s1_data, palette='viridis')

    plt.title('Scenariusz 1: Baseline (Trening i Test na tym samym zbiorze)', fontsize=15)
    plt.xlabel('Zbiór Danych (KDD / CORES / NETFLOW)', fontsize=12)
    plt.ylabel('F1-Score (Dystrybucja z K-Fold)', fontsize=12)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_s2_transfer_matrix(df):
    """
    Wykres dla S2: Jak modele radzą sobie na obcych danych?
    Używamy 'FacetGrid' aby pokazać pary Źródło -> Cel.
    """
    s2_data = df[df['Scenario'] == 'S2_Transfer']

    # Tworzymy kolumnę pomocniczą do opisu transferu
    s2_data['Transfer'] = s2_data['Train_Set'] + ' -> ' + s2_data['Test_Set']

    plt.figure(figsize=(14, 8))
    # Obracamy wykres (orient='h') dla czytelności długich etykiet
    sns.boxplot(x='F1-Score', y='Transfer', hue='Model', data=s2_data, palette='magma')

    plt.title('Scenariusz 2: Transfer Learning (Cross-Domain)', fontsize=15)
    plt.xlabel('F1-Score', fontsize=12)
    plt.ylabel('Transfer (Źródło -> Cel)', fontsize=12)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)  # Linia referencyjna
    plt.tight_layout()
    plt.show()


def plot_s3_combined_vs_baseline(df):
    """
    Wykres dla S3: Czy połączenie danych (Combined) pobiło Baseline (S1)?
    Porównujemy tylko najlepszy wynik dla danego zbioru testowego.
    """
    # Wybieramy tylko S1 i S3
    subset = df[df['Scenario'].isin(['S1_Baseline', 'S3_Combined'])].copy()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Test_Set', y='F1-Score', hue='Scenario', data=subset, palette='Set2')

    plt.title('S1 (Baseline) vs S3 (Combined Training)', fontsize=15)
    plt.xlabel('Zbiór Testowy (Target)', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.legend(title='Scenariusz')
    plt.tight_layout()
    plt.show()


def plot_model_stability(df):
    """
    Dodatkowy: Ogólna stabilność modeli (wariancja) niezależnie od danych.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y='F1-Score', data=df, palette='coolwarm')
    plt.title('Ogólna stabilność modeli we wszystkich eksperymentach', fontsize=15)
    plt.ylabel('F1-Score Distribution', fontsize=12)
    plt.show()


def main():
    df = load_results()
    if df is not None:
        # Ustawienie kolejności modeli dla spójności kolorów
        df = df.sort_values(by=['Model', 'Test_Set'])

        print("Generowanie wykresu dla S1...")
        plot_s1_baseline(df)

        print("Generowanie wykresu dla S2...")
        plot_s2_transfer_matrix(df)

        print("Generowanie porównania S1 vs S3...")
        plot_s3_combined_vs_baseline(df)

        print("Generowanie wykresu stabilności...")
        plot_model_stability(df)


if __name__ == "__main__":
    main()