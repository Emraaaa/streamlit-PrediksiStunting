import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
from sklearn.model_selection import train_test_split

# Grafik Stunting
def plot_stunting_graph(data):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x='Tahun',
        y='Persentase_Stunting',
        marker='o',
        color='blue',
        ax=ax
    )
    ax.set_title('Perkembangan Persentase Stunting (2001-2022)', fontsize=16)
    ax.set_xlabel('Tahun', fontsize=12)
    ax.set_ylabel('Persentase Stunting (%)', fontsize=12)
    return fig

# Grafik Ketidakcukupan Pangan
def plot_pangan_graph(data):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x='Tahun',
        y='PoU',
        marker='o',
        color='blue',
        ax=ax
    )
    ax.set_title('Perkembangan Ketidakcukupan Pangan', fontsize=16)
    ax.set_xlabel('Tahun', fontsize=12)
    ax.set_ylabel('Ketidakcukupan Pangan (%)', fontsize=12)
    return fig

# Heatmap
def heatmap(data):
    corr = data.corr() 
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))  
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Heatmap Korelasi", fontsize=16)
    return fig

# Grafik Data Pre Processing
def visualize_data_split(data, train_size=0.8, random_state=42):
    if not all(col in data.columns for col in ["Tahun", "Persentase_Stunting", "PoU"]):
        raise ValueError("Kolom 'Tahun', 'Persentase_Stunting', dan 'PoU' harus ada di dataframe.")
    
    train_data, test_data = train_test_split(data, train_size=train_size, random_state=random_state)
    
    train_percentage = len(train_data) / len(data) * 100
    test_percentage = len(test_data) / len(data) * 100
    
    split_data = pd.DataFrame({
        "Data Split": ["Train", "Test"],
        "Percentage": [train_percentage, test_percentage]
    })
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Data Split", y="Percentage", data=split_data, palette="pastel", ax=ax)
    ax.set_title(f"Data Split (Train {train_size*100:.0f}% - Test {100-train_size*100:.0f}%)", fontsize=14)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("Data Split", fontsize=12)
    ax.set_ylim(0, 100)
    plt.show()
    plt.tight_layout()

def plot_actual_vs_predicted_with_model(data, model, year_column, actual_column, feature_columns):
    # Validasi kolom
    if year_column not in data.columns or actual_column not in data.columns:
        raise ValueError(f"Kolom '{year_column}' atau '{actual_column}' tidak ditemukan dalam dataframe.")
    for col in feature_columns:
        if col not in data.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataframe.")
    
    # Membagi data menjadi data latih (80%) dan data uji (20%)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Melatih model dengan data latih
    model.fit(train_data[feature_columns], train_data[actual_column])
    
    # Melakukan prediksi pada seluruh data
    predicted_values = model.predict(data[feature_columns])
    data["Predicted"] = predicted_values
    
    # Melakukan prediksi pada data uji untuk evaluasi metrik
    predicted_test_values = model.predict(test_data[feature_columns])
    test_data["Predicted"] = predicted_test_values
    
    # Hitung MAE, MSE, dan skor model (R-squared) untuk data uji
    mae = mean_absolute_error(test_data[actual_column], test_data["Predicted"])
    mse = mean_squared_error(test_data[actual_column], test_data["Predicted"])
    score = model.score(test_data[feature_columns], test_data[actual_column])  
    
    # Plot line chart menggunakan Seaborn
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=data[year_column], y=data[actual_column], label="Actual", marker="o", color="blue")
    sns.lineplot(x=data[year_column], y=data["Predicted"], label="Predicted", marker="o", color="red")
    
    # Pengaturan grafik
    plt.title("Comparison of Actual and Predicted Stunting Percentage", fontsize=16)
    plt.xlabel("Tahun", fontsize=12)
    plt.ylabel("Persentase Stunting", fontsize=12)
    plt.legend(title="Data Type")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Tampilkan metrik
    st.write("**Metrics:**")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Model Score (R-squared): {score:.2f}")
