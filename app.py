"""Aplikasi Streamlit untuk demonstrasi model batik vs non-batik."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from pipeline_config import MODEL_FEATURES
from web_app.inference import (
    FEATURE_DESCRIPTIONS,
    FEATURE_DOMAINS,
    FEATURE_LABELS,
    decode_uploaded_image,
    load_model_bundle,
    model_feature_importance,
    predict_image,
)
from web_app.styles import inject_styles


st.set_page_config(
    page_title="BatikLens — Klasifikasi Batik",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles(st)


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return load_model_bundle()


def section_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="section-kicker">{html.escape(kicker)}</div>'
        f'<h2 class="section-title">{html.escape(title)}</h2>'
        f'<p class="section-copy">{html.escape(copy)}</p>',
        unsafe_allow_html=True,
    )


def format_feature(value: float) -> str:
    return f"{value:,.1f}" if abs(value) >= 100 else f"{value:.4f}"


def render_sidebar(bundle) -> None:
    with st.sidebar:
        st.markdown("## ◈ BatikLens")
        st.caption("Prototipe inferensi penelitian")
        st.divider()
        st.markdown("**Model formal**")
        st.markdown(f"### {bundle.model_name}")
        col_a, col_b = st.columns(2)
        col_a.metric("CV macro-F1", f"{bundle.cv_result['cv_f1_macro_mean']:.3f}")
        col_b.metric("Eksternal", f"{bundle.external_result['external_f1_macro']:.3f}")
        st.caption("Model dipilih hanya dari 5-fold CV, bukan dari uji eksternal.")
        st.divider()
        st.markdown("**Alur analisis**")
        st.markdown(
            "1. Validasi citra  \n"
            "2. Ekstraksi enam fitur  \n"
            f"3. Prediksi {bundle.external_result['model']}  \n"
            "4. Visualisasi hasil"
        )
        st.divider()
        st.markdown("**Batas penggunaan**")
        st.caption(
            "Aplikasi ini adalah demonstrasi penelitian. Hasil tidak menentukan "
            "keaslian, asal budaya, nilai ekonomi, atau legalitas suatu kain."
        )


try:
    bundle = get_model_bundle()
except Exception as error:
    st.error(f"Aplikasi belum siap: {error}")
    st.stop()

render_sidebar(bundle)

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">◈ Computer vision yang dapat dijelaskan</div>
      <h1>Baca pola. Pahami keputusan model.</h1>
      <p>
        Unggah citra kain dan ikuti bagaimana struktur tepi, tekstur, serta
        frekuensi pola diterjemahkan menjadi prediksi batik atau non-batik.
      </p>
      <div class="hero-badges">
        <span class="hero-badge">6 fitur interpretable</span>
        <span class="hero-badge">Model dipilih via 5-fold CV</span>
        <span class="hero-badge">Tanpa menyimpan unggahan</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

section_intro(
    "Mulai analisis",
    "Unggah satu citra kain",
    "Format JPG, JPEG, PNG, WEBP, atau BMP · maksimal 10 MB · minimal 32 × 32 piksel.",
)

with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Pilih atau jatuhkan citra di sini",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
        help="Citra diproses di memori dan tidak disalin ke dataset penelitian.",
    )

if uploaded_file is None:
    with st.container(border=True):
        st.markdown(
            """
            <div class="empty-state">
              <div class="empty-icon">⌁</div>
              <strong>Hasil analisis akan muncul di sini</strong><br>
              <span>Mulai dengan citra kain yang terang, tajam, dan memperlihatkan pola.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    section_intro(
        "Yang dianalisis",
        "Tiga sudut pandang terhadap pola",
        "Model merangkum struktur citra menjadi enam angka, bukan menghafal nama file atau subjenis.",
    )
    cols = st.columns(3)
    cards = [
        ("Edge / morfologi", "2 fitur", "Kompleksitas motif dan jumlah kontur kecil."),
        ("Tekstur", "3 fitur", "Entropi serta homogenitas GLCM dan entropi LBP."),
        ("Frekuensi", "1 fitur", "Rasio puncak FFT untuk periodisitas pola."),
    ]
    for column, (label, value, description) in zip(cols, cards, strict=True):
        with column:
            st.markdown(
                f'<div class="metric-tile"><div class="label">{label}</div>'
                f'<div class="value">{value}</div><div class="desc">{description}</div></div>',
                unsafe_allow_html=True,
            )
    st.stop()

try:
    image_bytes = uploaded_file.getvalue()
    with st.spinner("Membaca pola dan mengekstrak enam fitur…"):
        bgr = decode_uploaded_image(image_bytes)
        result = predict_image(bgr, bundle)
except ValueError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"Analisis gagal dilakukan: {error}")
    st.stop()

display_name = uploaded_file.name
height, width = bgr.shape[:2]
st.caption(f"Berkas: {display_name} · {width} × {height} piksel")

section_intro(
    "Ringkasan keputusan",
    "Hasil klasifikasi",
    "Keputusan menggunakan ambang skor batik 0,50 pada model formal hasil cross-validation.",
)

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.image(result.visualizations["original"], caption="Citra yang dianalisis", width="stretch")

with right:
    verdict = "BATIK" if result.predicted_label == 1 else "NON-BATIK"
    card_class = "" if result.predicted_label == 1 else " non-batik"
    score_percent = result.score_batik * 100
    st.markdown(
        f"""
        <div class="result-card{card_class}">
          <div class="result-label">HASIL MODEL</div>
          <div class="result-value">{verdict}</div>
          <div class="result-score">Skor kelas terpilih: {result.confidence * 100:.1f}%</div>
          <div class="score-track"><div class="score-fill" style="width:{score_percent:.1f}%"></div></div>
          <div class="score-axis"><span>Non-batik</span><span>Skor batik {score_percent:.1f}%</span><span>Batik</span></div>
          <div class="model-note">Skor merupakan keluaran model dan belum dikalibrasi sebagai probabilitas statistik.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Gunakan hasil bersama inspeksi ahli. Model hanya membaca karakter visual "
        "yang dipelajari dari dataset penelitian.",
        icon="ℹ️",
    )

tab_process, tab_features, tab_model = st.tabs(
    ["Proses citra", "Enam fitur", "Tentang model"]
)

with tab_process:
    section_intro(
        "Jejak visual",
        "Dari gambar ke representasi pola",
        "Tiga visual berikut membantu membaca sinyal yang diringkas oleh fitur model.",
    )
    process_columns = st.columns(3, gap="medium")
    process_items = [
        (
            "motif_edge",
            "1 · Motif edge",
            "Tepi halus setelah struktur kasar disaring; dasar fitur morfologi.",
        ),
        (
            "lbp",
            "2 · Tekstur LBP",
            "Kode pola mikro lokal yang merangkum variasi permukaan.",
        ),
        (
            "fft",
            "3 · Spektrum FFT",
            "Energi frekuensi untuk menangkap periodisitas atau pengulangan pola.",
        ),
    ]
    for column, (key, title, caption) in zip(process_columns, process_items, strict=True):
        with column:
            with st.container(border=True):
                st.image(result.visualizations[key], width="stretch")
                st.markdown(f"**{title}**")
                st.caption(caption)

with tab_features:
    section_intro(
        "Representasi numerik",
        "Enam nilai yang masuk ke model",
        "Urutan dan definisinya identik dengan pipeline eksperimen pada paper.",
    )
    feature_columns = st.columns(3)
    for index, feature in enumerate(MODEL_FEATURES):
        with feature_columns[index % 3]:
            st.markdown(
                f'<div class="metric-tile"><div class="label">{FEATURE_DOMAINS[feature]}</div>'
                f'<div class="value">{format_feature(result.features[feature])}</div>'
                f'<div class="desc"><strong>{FEATURE_LABELS[feature]}</strong><br>'
                f'{FEATURE_DESCRIPTIONS[feature]}</div></div>',
                unsafe_allow_html=True,
            )
            st.write("")

    feature_frame = pd.DataFrame(
        {
            "Fitur": [FEATURE_LABELS[name] for name in MODEL_FEATURES],
            "Domain": [FEATURE_DOMAINS[name] for name in MODEL_FEATURES],
            "Nilai": [result.features[name] for name in MODEL_FEATURES],
        }
    )
    with st.expander("Lihat tabel nilai mentah"):
        st.dataframe(feature_frame, hide_index=True, width="stretch")

with tab_model:
    section_intro(
        "Provenance model",
        "Terikat pada pipeline penelitian",
        "Aplikasi memuat artefak final yang dipilih berdasarkan performa cross-validation.",
    )
    metric_cols = st.columns(4)
    metrics = [
        ("Model", bundle.model_name),
        ("CV macro-F1", f"{bundle.cv_result['cv_f1_macro_mean']:.3f}"),
        ("CV simpangan", f"± {bundle.cv_result['cv_f1_macro_std']:.3f}"),
        ("Uji eksternal", f"{bundle.external_result['external_f1_macro']:.3f}"),
    ]
    for column, (label, value) in zip(metric_cols, metrics, strict=True):
        column.metric(label, value)

    importance = model_feature_importance(bundle)
    if importance:
        importance_frame = pd.DataFrame(
            {
                "Fitur": [FEATURE_LABELS[name] for name in MODEL_FEATURES],
                "Importance global": [importance[name] for name in MODEL_FEATURES],
            }
        ).set_index("Fitur")
        st.markdown("#### Importance fitur global")
        st.bar_chart(importance_frame, horizontal=True, color="#a84d32")
        st.caption(
            f"Importance bersifat global pada {bundle.external_result['model']} dan bukan "
            "penjelasan kausal untuk satu citra tertentu."
        )

st.divider()
st.markdown(
    "<p class='tiny-note'><strong>BatikLens</strong> · Demonstrasi klasifikasi biner "
    "berbasis enam fitur interpretable. Unggahan diproses dalam memori sesi dan tidak "
    "ditambahkan ke dataset.</p>",
    unsafe_allow_html=True,
)
