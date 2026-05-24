import base64
import pickle

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title='PreSisi',
    page_icon='logo_presisi.png',
    layout='wide',
)

# Load model pipeline terbaik dari artifact skenario 3.
with open('best_model_skenario_3_artifacts.sav', 'rb') as model_file:
    model_artifacts = pickle.load(model_file)
    cirrhosis_model = model_artifacts['model']

# Daftar fitur harus sama dengan kolom yang digunakan saat training model.
FEATURE_COLUMNS = [
    'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',
    'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage',
    'Drug_D-penicillamine', 'Drug_Placebo', 'Sex_F', 'Sex_M',
    'Ascites_N', 'Ascites_Y', 'Hepatomegaly_N', 'Hepatomegaly_Y',
    'Spiders_N', 'Spiders_Y', 'Edema_N', 'Edema_S', 'Edema_Y',
]

NUMERIC_COLUMNS = [
    'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',
    'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage',
]

# Kolom kategori sudah diubah menjadi one-hot encoding.
OHE_COLUMNS = [column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS]

# Mapping output model agar hasil prediksi lebih mudah dibaca.
CLASS_LABELS = {
    0: 'Death',
    1: 'Censored',
}

SEX_LABELS = {
    'F': 'Female',
    'M': 'Male',
}

YES_NO_LABELS = {
    'N': 'No',
    'Y': 'Yes',
}

EDEMA_LABELS = {
    'N': 'N (no edema and no diuretic therapy for edema)',
    'S': 'S (edema present without diuretics, or edema resolved by diuretics)',
    'Y': 'Y (edema despite diuretic therapy)',
}

LOGO_PATH = 'logo_presisi.png'
STYLE_PATH = 'styles.css'


def image_to_base64(path):
    # Mengubah logo menjadi base64 agar bisa ditampilkan dalam HTML Streamlit.
    with open(path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_css(path):
    # Memuat file CSS eksternal untuk styling tampilan aplikasi.
    with open(path, 'r') as style_file:
        css = style_file.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


load_css(STYLE_PATH)


def create_input_data(
    age, bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
    tryglicerides, platelets, prothrombin, stage, drug, sex, ascites,
    hepatomegaly, spiders, edema,
):
    # Mengubah input user menjadi DataFrame dengan urutan kolom sesuai model.
    return pd.DataFrame([{
        'Age': float(age),
        'Bilirubin': float(bilirubin),
        'Cholesterol': float(cholesterol),
        'Albumin': float(albumin),
        'Copper': float(copper),
        'Alk_Phos': float(alk_phos),
        'SGOT': float(sgot),
        'Tryglicerides': float(tryglicerides),
        'Platelets': float(platelets),
        'Prothrombin': float(prothrombin),
        'Stage': float(stage),
        'Drug_D-penicillamine': int(drug == 'D-penicillamine'),
        'Drug_Placebo': int(drug == 'Placebo'),
        'Sex_F': int(sex == 'F'),
        'Sex_M': int(sex == 'M'),
        'Ascites_N': int(ascites == 'N'),
        'Ascites_Y': int(ascites == 'Y'),
        'Hepatomegaly_N': int(hepatomegaly == 'N'),
        'Hepatomegaly_Y': int(hepatomegaly == 'Y'),
        'Spiders_N': int(spiders == 'N'),
        'Spiders_Y': int(spiders == 'Y'),
        'Edema_N': int(edema == 'N'),
        'Edema_S': int(edema == 'S'),
        'Edema_Y': int(edema == 'Y'),
    }], columns=FEATURE_COLUMNS)


def class_label(class_value):
    # Mengubah nilai class dari model menjadi label yang ditampilkan di UI.
    return CLASS_LABELS.get(class_value, CLASS_LABELS.get(str(class_value), str(class_value)))


def build_readable_input(
    age, bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
    tryglicerides, platelets, prothrombin, stage, drug, sex, ascites,
    hepatomegaly, spiders, edema,
):
    # Membuat ringkasan input user dalam format yang mudah dibaca.
    return pd.DataFrame({
        'Variable': [
            'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',
            'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage',
            'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema',
        ],
        'User Input': [
            age, bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
            tryglicerides, platelets, prothrombin, stage, drug,
            SEX_LABELS[sex], YES_NO_LABELS[ascites], YES_NO_LABELS[hepatomegaly],
            YES_NO_LABELS[spiders], EDEMA_LABELS[edema],
        ],
    })


def build_ohe_table(raw_input_data):
    # Menampilkan nilai hasil one-hot encoding untuk fitur kategori.
    return pd.DataFrame({
        'One-Hot Encoding Column': OHE_COLUMNS,
        'Value': raw_input_data[OHE_COLUMNS].iloc[0].values,
    })


def build_number_column_config(columns, width='medium', number_format='%.2f'):
    # Mengatur lebar dan format kolom angka pada tabel Streamlit.
    return {
        column: st.column_config.NumberColumn(
            column,
            width=width,
            format=number_format,
        )
        for column in columns
    }


def find_pipeline_step(estimator, class_name):
    # Mencari step tertentu di dalam pipeline model.
    if estimator.__class__.__name__ == class_name:
        return estimator

    for _, step in getattr(estimator, 'steps', []):
        found_step = find_pipeline_step(step, class_name)
        if found_step is not None:
            return found_step

    for transformer in getattr(estimator, 'transformers_', []):
        if len(transformer) >= 2:
            found_step = find_pipeline_step(transformer[1], class_name)
            if found_step is not None:
                return found_step

    return None


def build_standard_scaler_table(model, raw_input_data):
    # Membuat tabel hasil StandardScaler dari pipeline untuk kebutuhan transparansi.
    standard_scaler = find_pipeline_step(model, 'StandardScaler')
    if standard_scaler is None:
        return None

    scaler_columns = list(getattr(standard_scaler, 'feature_names_in_', raw_input_data.columns))
    scaler_input = raw_input_data[scaler_columns]
    scaled_values = standard_scaler.transform(scaler_input)
    scaled_by_column = pd.DataFrame(scaled_values, columns=scaler_columns)

    scaled_input_data = raw_input_data.copy()
    scaled_input_data[NUMERIC_COLUMNS] = scaled_by_column[NUMERIC_COLUMNS].values

    return scaled_input_data


def build_probability_table(model, raw_input_data):
    # Mengambil probabilitas prediksi untuk setiap class jika model mendukung predict_proba.
    if not hasattr(model, 'predict_proba'):
        return None

    probabilities = model.predict_proba(raw_input_data)[0]
    classes = getattr(model, 'classes_', range(len(probabilities)))

    return pd.DataFrame({
        'Class': [class_label(value) for value in classes],
        'Probability': probabilities,
        'Probability (%)': probabilities * 100,
    })


def show_prediction_result(prediction, probability_table):
    # Menampilkan hasil prediksi utama dan probabilitas class.
    result_label = class_label(prediction)

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">Prediction Result</div>
            <div class="prediction-value">{result_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probability_table is not None:
        probability_items = []
        ordered_probability_table = probability_table.sort_values(
            by='Probability (%)',
            ascending=False,
        )
        for _, row in ordered_probability_table.iterrows():
            probability_items.append(
                f'<span class="probability-item">'
                f'<strong>{row["Probability (%)"]:.0f}%</strong> {row["Class"]}'
                f'</span>'
            )
        st.markdown(
            f"""
            <div class="probability-card">
                <div class="prediction-label">Prediction Probability</div>
                <div class="probability-list">{''.join(probability_items)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# def show_prediction_details(
#     readable_input, raw_input_data, scaler_table, probability_table,
# ):
#     st.subheader('Prediction Transparency Details')
#     st.caption(
#         'This section shows how the original user input is processed before entering the model: '
#         'categorical data is converted with One-Hot Encoding, while StandardScaler and oversampling '
#         'are already included inside the model pipeline.'
#     )

#     tab_summary, tab_scaler, tab_ohe, tab_model = st.tabs([
#         'Input Summary',
#         'StandardScaler',
#         'One-Hot Encoding',
#         'Model Info',
#     ])

#     with tab_summary:
#         st.markdown('**Original user input**')
#         st.dataframe(readable_input, use_container_width=True, hide_index=True)

#     with tab_scaler:
#         st.markdown('**StandardScaler result from the model pipeline**')
#         if scaler_table is None:
#             st.info('StandardScaler was not found inside the loaded model pipeline.')
#         else:
#             st.dataframe(
#                 scaler_table.style.format('{:.4f}'),
#                 use_container_width=True,
#                 hide_index=True,
#             )
#             st.info(
#                 'Only numeric columns are scaled. One-Hot Encoding columns remain 0 or 1. '
#                 'Prediction still uses the full pipeline directly.'
#             )

#     with tab_ohe:
#         st.markdown('**One-Hot Encoding result for categorical data**')
#         st.dataframe(
#             build_ohe_table(raw_input_data),
#             use_container_width=True,
#             hide_index=True,
#         )
#         st.info('Value 1 means the category is selected, while 0 means it is not selected.')

#     with tab_model:
#         st.markdown('**Final data entered into the model**')
#         st.dataframe(
#             raw_input_data,
#             column_config=build_number_column_config(raw_input_data.columns, width='medium'),
#             use_container_width=True,
#             hide_index=True,
#         )
#         st.info(
#             'This model uses a pipeline, so StandardScaler and oversampling '
#             'are applied internally by the model.'
#         )

#         if probability_table is not None:
#             st.markdown('**Prediction probability per class**')
#             st.dataframe(
#                 probability_table.style.format({
#                     'Probability': '{:.4f}',
#                     'Probability (%)': '{:.2f}',
#                 }),
#                 use_container_width=True,
#                 hide_index=True,
#             )


logo_base64 = image_to_base64(LOGO_PATH)
# Header aplikasi berisi logo dan judul.
st.markdown(
    f"""
    <div class="app-navbar">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo PreSisi">
        <h2 class="app-title">PreSisi: Prediksi Survival Pasien Sirosis Hati</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

_, form_area, _ = st.columns([0.8, 4.4, 0.8])

with form_area:
    with st.form('predict_form'):
        # Form input fitur pasien sesuai variabel dataset.
        col1, col2 = st.columns(2, gap='medium')

        with col1:
            age = st.number_input('Age (days)', min_value=0, value=0, step=1)
            cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=0.0, value=0.0, step=1.0)
            copper = st.number_input('Copper (ug/day)', min_value=0.0, value=0.0, step=1.0)
            sgot = st.number_input('SGOT (U/ml)', min_value=0.0, value=0.0, step=1.0)
            platelets = st.number_input('Platelets (ml/1000)', min_value=0.0, value=0.0, step=1.0)
            stage = st.selectbox('Stage', [1, 2, 3, 4])
            sex = st.selectbox('Sex', list(SEX_LABELS), format_func=SEX_LABELS.get)
            hepatomegaly = st.selectbox('Hepatomegaly', list(YES_NO_LABELS), format_func=YES_NO_LABELS.get)
            edema = st.selectbox('Edema', list(EDEMA_LABELS), format_func=EDEMA_LABELS.get)

        with col2:
            bilirubin = st.number_input('Bilirubin (mg/dl)', min_value=0.0, value=0.0, step=0.1)
            albumin = st.number_input('Albumin (gm/dl)', min_value=0.0, value=0.0, step=0.1)
            alk_phos = st.number_input('Alk_Phos (U/liter)', min_value=0.0, value=0.0, step=1.0)
            tryglicerides = st.number_input('Tryglicerides (mg/dl)', min_value=0.0, value=0.0, step=1.0)
            prothrombin = st.number_input('Prothrombin (s)', min_value=0.0, value=0.0, step=0.1)
            drug = st.selectbox('Drug', ['D-penicillamine', 'Placebo'])
            ascites = st.selectbox('Ascites', list(YES_NO_LABELS), format_func=YES_NO_LABELS.get)
            spiders = st.selectbox('Spiders', list(YES_NO_LABELS), format_func=YES_NO_LABELS.get)

        st.markdown('')
        predict_button = st.form_submit_button('Predict', use_container_width=False)

if predict_button:
    # Membuat data input, menjalankan prediksi, lalu menampilkan hasilnya.
    raw_input_data = create_input_data(
        age, bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
        tryglicerides, platelets, prothrombin, stage, drug, sex, ascites,
        hepatomegaly, spiders, edema,
    )
    prediction = cirrhosis_model.predict(raw_input_data)[0]
    probability_table = build_probability_table(cirrhosis_model, raw_input_data)
    # scaler_table = build_standard_scaler_table(cirrhosis_model, raw_input_data)
    readable_input = build_readable_input(
        age, bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
        tryglicerides, platelets, prothrombin, stage, drug, sex, ascites,
        hepatomegaly, spiders, edema,
    )

    _, result_area, _ = st.columns([0.8, 4.4, 0.8])
    with result_area:
        show_prediction_result(prediction, probability_table)
        # show_prediction_details(
        #     readable_input,
        #     raw_input_data,
        #     scaler_table,
        #     probability_table,
        # )

st.markdown(
    # Footer custom aplikasi.
    """
    <footer class="app-footer">
        &copy; 2026 <strong>PreSisi</strong> by dwicahyanovita
    </footer>
    """,
    unsafe_allow_html=True,
)
