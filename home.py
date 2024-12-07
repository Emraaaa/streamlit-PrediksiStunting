import pickle
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from fungsi_grafik import plot_stunting_graph, plot_pangan_graph, heatmap, visualize_data_split, plot_actual_vs_predicted_with_model

# Set page layout to wide
st.set_page_config(layout="wide")

# Load model
model = pickle.load(open('model', 'rb'))

# Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",  
        options=["Grafik", "Prediksi Stunting"],  
        icons=["graph-up", "bar-chart-line"],  
        menu_icon="menu-app",  
        default_index=0,  
        styles={
            "container": {"padding": "5px", "background-color": "#f8f9fa"},
            "icon": {"color": "blue", "font-size": "20px"}, 
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#e9ecef",
            },
            "nav-link-selected": {"background-color": "#007bff", "color": "white"},
        },
    )
    st.sidebar.markdown('''---\nCreated by Apex Predator.\n''')

# Judul 
st.title('Dashboard Prediksi Persentase Stunting')

# Fungsi untuk membaca data CSV
@st.cache_data
def load_csv_data(file_path):
    return pd.read_csv(file_path)

# Fungsi untuk membaca data Excel
@st.cache_data
def load_excel_data(file_path):
    return pd.read_excel(file_path)

# Membaca data
data_stunting_csv = "data_stunting.csv"
df_stunting = load_csv_data(data_stunting_csv)

data_file_2_csv = "5F8A486_ALL_LATEST.csv"
df_stunting_2 = load_csv_data(data_file_2_csv)

data_file_excel = "nasional per tahun.xlsx"
df_pangan = load_excel_data(data_file_excel)

# Tampilan Prediksi
if selected == "Prediksi Stunting":
    st.subheader("Prediksi Persentase Stunting")
    year = st.number_input('Ingin Prediksi Tahun Berapa?', min_value=2023, max_value=2100, step=1)
    PoU = st.number_input('Perkiraan Persentase Ketidakcukupan Pangan di Tahun Tersebut', min_value=0.0, max_value=100.0, step=0.1)

    if st.button('Prediksi'):
        try:
            predict = model.predict([[year, PoU]])
            st.write(f'Perkiraan Persentase Stunting pada tahun {year} adalah **{predict[0]:.2f}%**')
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# Tampilan Grafik
elif selected == "Grafik":

    # Membuat dua kolom untuk Data dan Grafik
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Data")
        df_stunting['Tahun'] = df_stunting['Tahun'].astype(str)
        st.write("Dataset Loaded Successfully:", df_stunting)

    with col2:
        st.subheader("Grafik")
        graph_option = st.selectbox(
            "Pilih Grafik yang Ingin Ditampilkan:",
            ["Grafik Stunting", "Grafik Ketidakcukupan Pangan"]
        )

        # Grafik Stunting
        if graph_option == "Grafik Stunting":
            try:
                df_stunting_2 = df_stunting_2[df_stunting_2['GEO_NAME_SHORT'].str.contains('Indonesia', na=False)]
                df_stunting_2 = df_stunting_2.sort_values(by='DIM_TIME', ascending=True)
                df_stunting_2 = df_stunting_2[['DIM_TIME', 'RATE_PER_100_N']].reset_index(drop=True)
                df_stunting_2 = df_stunting_2.rename(columns={'DIM_TIME': 'Tahun', 'RATE_PER_100_N': 'Persentase_Stunting'})

                fig = plot_stunting_graph(df_stunting_2)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Gagal membaca data: {e}")

        # Grafik Ketidakcukupan Pangan
        elif graph_option == "Grafik Ketidakcukupan Pangan":
            try:
                fig = plot_pangan_graph(df_pangan)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Gagal membaca data: {e}")

    # Korelasi dan Data Pre-Processing
    col3, col4 = st.columns([1, 1])
    with col3:
        st.subheader("Korelasi")
        try:
            fig = heatmap(df_stunting)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal membuat heatmap: {e}")

    with col4:
        st.subheader("Data Pre-Processing")
        try:
            fig = visualize_data_split(df_stunting, train_size=0.8, random_state=42)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal memproses data atau membuat visualisasi: {e}")

    # Grafik Aktual vs Model
    st.subheader("Grafik Aktual VS Model")
    try:
        fig = plot_actual_vs_predicted_with_model(
            data=df_stunting,
            model=model,
            year_column="Tahun",
            actual_column="Persentase_Stunting",
            feature_columns=["Tahun", "PoU"]
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Gagal membuat grafik aktual vs model: {e}")
