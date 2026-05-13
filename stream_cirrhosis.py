import base64
import pickle

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title='PreSisi',
    page_icon='logo_presisi.png',
    layout='wide',
)

with open('random_forest_model.sav', 'rb') as model_file:
    cirrhosis_model = pickle.load(model_file)

with open('standard_scaler.sav', 'rb') as scaler_file:
    standard_scaler = pickle.load(scaler_file)

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

OHE_COLUMNS = [column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS]

CLASS_LABELS = {
    0: 'Meninggal dunia',
    1: 'Bertahan hidup',
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
    'N': 'Tidak ada edema dan tidak ada terapi diuretik (N)',
    'S': 'Edema tanpa diuretik atau membaik dengan diuretik (S)',
    'Y': 'Edema tetap ada meskipun dengan terapi diuretik (Y)',
}

LOGO_PATH = 'logo_presisi.png'
STYLE_PATH = 'styles.css'


def image_to_base64(path):
    with open(path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_css(path):
    with open(path, 'r') as style_file:
        css = style_file.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


load_css(STYLE_PATH)


def create_input_data(
    age, bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
    tryglicerides, platelets, prothrombin, stage, drug, sex, ascites,
    hepatomegaly, spiders, edema,
):
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


def scale_input_data(input_data):
    scaled_data = input_data.copy()
    scaled_data[NUMERIC_COLUMNS] = standard_scaler.transform(scaled_data[NUMERIC_COLUMNS])
    return scaled_data


def class_label(class_value):
    return CLASS_LABELS.get(int(class_value), str(class_value))


def build_readable_input(
    age, bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
    tryglicerides, platelets, prothrombin, stage, drug, sex, ascites,
    hepatomegaly, spiders, edema,
):
    return pd.DataFrame({
        'Variabel': [
            'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',
            'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage',
            'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema',
        ],
        'Input User': [
            age, bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
            tryglicerides, platelets, prothrombin, stage, drug,
            SEX_LABELS[sex], YES_NO_LABELS[ascites], YES_NO_LABELS[hepatomegaly],
            YES_NO_LABELS[spiders], EDEMA_LABELS[edema],
        ],
    })


def build_scaler_table(raw_input_data, scaled_input_data):
    return pd.DataFrame({
        'Kolom Numerik': NUMERIC_COLUMNS,
        'Nilai Asli': raw_input_data[NUMERIC_COLUMNS].iloc[0].values,
        'Hasil StandardScaler': scaled_input_data[NUMERIC_COLUMNS].iloc[0].values,
    })


def build_ohe_table(raw_input_data):
    return pd.DataFrame({
        'Kolom One-Hot Encoding': OHE_COLUMNS,
        'Nilai': raw_input_data[OHE_COLUMNS].iloc[0].values,
    })


def build_probability_table(model, scaled_input_data):
    if not hasattr(model, 'predict_proba'):
        return None

    probabilities = model.predict_proba(scaled_input_data)[0]
    classes = getattr(model, 'classes_', range(len(probabilities)))

    return pd.DataFrame({
        'Kelas': [class_label(value) for value in classes],
        'Probabilitas': probabilities,
        'Probabilitas (%)': probabilities * 100,
    })


def show_prediction_result(prediction, probability_table):
    result_label = class_label(prediction)

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">Hasil Prediksi</div>
            <div class="prediction-value">{result_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probability_table is not None:
        selected = probability_table[probability_table['Kelas'] == result_label]
        if not selected.empty:
            confidence = selected.iloc[0]['Probabilitas (%)']
            st.metric('Keyakinan model untuk hasil ini', f'{confidence:.2f}%')


def show_prediction_details(
    readable_input, raw_input_data, scaled_input_data, probability_table,
):
    st.subheader('Detail Transparansi Prediksi')
    st.caption(
        'Bagian ini menunjukkan bagaimana input asli dari user diproses sebelum masuk ke model: '
        'data kategori diubah dengan One-Hot Encoding, lalu kolom numerik diubah dengan StandardScaler.'
    )

    tab_summary, tab_scaler, tab_ohe, tab_model = st.tabs([
        'Ringkasan Input',
        'StandardScaler',
        'One-Hot Encoding',
        'Info Model',
    ])

    with tab_summary:
        st.markdown('**Input asli dari user**')
        st.dataframe(readable_input, use_container_width=True, hide_index=True)

    with tab_scaler:
        st.markdown('**Kolom numerik sebelum dan sesudah StandardScaler**')
        st.dataframe(
            build_scaler_table(raw_input_data, scaled_input_data).style.format({
                'Nilai Asli': '{:.4f}',
                'Hasil StandardScaler': '{:.4f}',
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            'StandardScaler memakai nilai rata-rata dan standar deviasi dari data training. '
            'Karena itu user tetap mengisi data asli, tetapi model menerima data numerik yang sudah diskalakan.'
        )

    with tab_ohe:
        st.markdown('**Hasil One-Hot Encoding untuk data kategori**')
        st.dataframe(
            build_ohe_table(raw_input_data),
            use_container_width=True,
            hide_index=True,
        )
        st.info('Nilai 1 berarti kategori tersebut dipilih, sedangkan 0 berarti tidak dipilih.')

    with tab_model:
        st.markdown('**Data final yang masuk ke model**')
        st.dataframe(
            scaled_input_data.style.format('{:.4f}'),
            use_container_width=True,
            hide_index=True,
        )

        if probability_table is not None:
            st.markdown('**Probabilitas prediksi per kelas**')
            st.dataframe(
                probability_table.style.format({
                    'Probabilitas': '{:.4f}',
                    'Probabilitas (%)': '{:.2f}',
                }),
                use_container_width=True,
                hide_index=True,
            )


logo_base64 = image_to_base64(LOGO_PATH)
st.markdown(
    f"""
    <div class="app-navbar">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo PreSisi">
        <h1 class="app-title">PreSisi: Prediksi Survival Pasien Sirosis Hati</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

_, form_area, _ = st.columns([0.8, 4.4, 0.8])

with form_area:
    with st.form('predict_form'):
        col1, col2 = st.columns(2, gap='medium')

        with col1:
            age = st.number_input('Usia pasien (hari)', min_value=0, value=0, step=1)
            cholesterol = st.number_input('Kolesterol (mg/dl)', min_value=0.0, value=0.0, step=1.0)
            copper = st.number_input('Copper (ug/hari)', min_value=0.0, value=0.0, step=1.0)
            sgot = st.number_input('SGOT (U/ml)', min_value=0.0, value=0.0, step=1.0)
            platelets = st.number_input('Trombosit (ml/1000)', min_value=0.0, value=0.0, step=1.0)
            stage = st.selectbox('Stadium', [1, 2, 3, 4])
            sex = st.selectbox('Jenis kelamin', list(SEX_LABELS), format_func=SEX_LABELS.get)
            hepatomegaly = st.selectbox('Hepatomegaly', list(YES_NO_LABELS), format_func=YES_NO_LABELS.get)
            edema = st.selectbox('Edema', list(EDEMA_LABELS), format_func=EDEMA_LABELS.get)

        with col2:
            bilirubin = st.number_input('Bilirubin (mg/dl)', min_value=0.0, value=0.0, step=0.1)
            albumin = st.number_input('Albumin (gm/dl)', min_value=0.0, value=0.0, step=0.1)
            alk_phos = st.number_input('Alkaline Phosphatase (U/liter)', min_value=0.0, value=0.0, step=1.0)
            tryglicerides = st.number_input('Trigliserida', min_value=0.0, value=0.0, step=1.0)
            prothrombin = st.number_input('Protrombin (detik)', min_value=0.0, value=0.0, step=0.1)
            drug = st.selectbox('Drug', ['D-penicillamine', 'Placebo'])
            ascites = st.selectbox('Ascites', list(YES_NO_LABELS), format_func=YES_NO_LABELS.get)
            spiders = st.selectbox('Spiders', list(YES_NO_LABELS), format_func=YES_NO_LABELS.get)

        st.markdown('')
        predict_button = st.form_submit_button('Prediksi', use_container_width=False)

if predict_button:
    raw_input_data = create_input_data(
        age, bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
        tryglicerides, platelets, prothrombin, stage, drug, sex, ascites,
        hepatomegaly, spiders, edema,
    )
    scaled_input_data = scale_input_data(raw_input_data)
    prediction = cirrhosis_model.predict(scaled_input_data)[0]
    probability_table = build_probability_table(cirrhosis_model, scaled_input_data)
    readable_input = build_readable_input(
        age, bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
        tryglicerides, platelets, prothrombin, stage, drug, sex, ascites,
        hepatomegaly, spiders, edema,
    )

    _, result_area, _ = st.columns([0.8, 4.4, 0.8])
    with result_area:
        show_prediction_result(prediction, probability_table)
        show_prediction_details(
            readable_input,
            raw_input_data,
            scaled_input_data,
            probability_table,
        )

st.markdown(
    """
    <footer class="app-footer">
        &copy; 2026 <strong>PreSisi</strong> by dwicahyanovita
    </footer>
    """,
    unsafe_allow_html=True,
)
