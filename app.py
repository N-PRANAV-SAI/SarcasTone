#!/usr/bin/env python3
"""
SarcasTone - Multimodal Sarcasm Detection Using Text and Speech Prosody
Streamlit Application
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sarcastone.config import *
from sarcastone.features.sentiment_analyzer import (
    analyze_text_sentiment, analyze_acoustic_tone, detect_sentiment_tone_mismatch
)

# ─── Page Config ───
st.set_page_config(
    page_title="SarcasTone – Sarcasm Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Cached Model Loading ───
@st.cache_resource
def load_all_models():
    """Load all trained models once and cache them."""
    import gc
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    models = {}

    # Text model
    text_model_path = os.path.join(MODELS_DIR, 'text_model')
    has_text_model = os.path.exists(text_model_path) and any(
        p.startswith('model.pt') for p in os.listdir(text_model_path)
    )
    if has_text_model:
        try:
            from sarcastone.models.text_model import load_text_model
            model, tokenizer = load_text_model(device='cpu')
            models['text'] = {'model': model, 'tokenizer': tokenizer}
        except Exception as e:
            st.warning(f"Could not load text model: {e}")

    # Audio model
    audio_model_path = os.path.join(MODELS_DIR, 'audio_model.joblib')
    if os.path.exists(audio_model_path):
        try:
            from sarcastone.models.audio_model import load_audio_model
            models['audio'] = load_audio_model()
        except Exception as e:
            st.warning(f"Could not load audio model: {e}")

    # Multimodal model
    mm_model_path = os.path.join(MODELS_DIR, 'multimodal_model.joblib')
    if os.path.exists(mm_model_path):
        try:
            from sarcastone.models.multimodal_model import load_multimodal_model
            models['multimodal'] = load_multimodal_model()
        except Exception as e:
            st.warning(f"Could not load multimodal model: {e}")

    gc.collect()
    return models


@st.cache_data
def load_evaluation_results():
    """Load saved evaluation results."""
    results = {}
    for name in ['text_model', 'audio_model', 'multimodal_model', 'model_comparison']:
        path = os.path.join(RESULTS_DIR, f'{name}_results.json')
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
    return results


@st.cache_data
def load_dataset_info():
    """Load dataset for display."""
    from sarcastone.data_utils import load_dataset
    return load_dataset()


@st.cache_data
def get_sample_manifest():
    """Load 10 pre-generated sample audio and multimodal clips."""
    manifest_path = os.path.join(PROJECT_ROOT, "data", "sample_audios", "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return []


# ─── Helper Functions ───
def create_gauge_chart(value, title, color_ranges=None):
    """Create a gauge chart for confidence display."""
    if color_ranges is None:
        color_ranges = [
            [0, 0.4, "red"],
            [0.4, 0.7, "orange"],
            [0.7, 1.0, "green"]
        ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 16}},
        number={'suffix': '%', 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [r[0]*100, r[1]*100], 'color': r[2]} for r in color_ranges
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_confusion_matrix_fig(cm, title="Confusion Matrix"):
    """Create a plotly confusion matrix heatmap."""
    labels = ['Not Sarcastic', 'Sarcastic']
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels, y=labels,
        color_continuous_scale='Blues',
        text_auto=True
    )
    fig.update_layout(title=title, height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def predict_text(text, models):
    """Get text model prediction."""
    if 'text' not in models:
        return None, None
    import torch
    from sarcastone.models.text_model import TextSarcasmClassifier

    model = models['text']['model']
    tokenizer = models['text']['tokenizer']

    encoding = tokenizer(
        str(text), add_special_tokens=True, max_length=TEXT_MAX_LENGTH,
        padding='max_length', truncation=True, return_tensors='pt'
    )

    model.eval()
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    prediction = int(np.argmax(probs))
    confidence = float(probs[prediction])
    return prediction, confidence


def predict_audio_file(audio_path, models):
    """Get audio model prediction from file."""
    if 'audio' not in models:
        return None, None
    from sarcastone.features.prosody_analyzer import extract_audio_features_for_model
    from sarcastone.models.audio_model import predict_audio

    features = extract_audio_features_for_model(audio_path)
    return predict_audio(models['audio'], features)


def predict_multimodal_input(text, audio_path, models):
    """Get multimodal prediction."""
    if 'multimodal' not in models or 'text' not in models:
        return None, None
    import torch
    from sarcastone.features.prosody_analyzer import extract_audio_features_for_model
    from sarcastone.models.multimodal_model import predict_multimodal

    # Get text embedding
    model = models['text']['model']
    tokenizer = models['text']['tokenizer']
    encoding = tokenizer(
        str(text), add_special_tokens=True, max_length=TEXT_MAX_LENGTH,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    model.eval()
    with torch.no_grad():
        outputs = model.bert(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
        text_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()

    # Get audio features
    audio_feat = extract_audio_features_for_model(audio_path)

    return predict_multimodal(models['multimodal'], text_emb, audio_feat)


def transcribe_audio(audio_path):
    """Transcribe audio using speech recognition."""
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        return f"[Transcription unavailable: {e}]"


# ─── Sidebar ───
st.sidebar.title("🎭 SarcasTone")
st.sidebar.markdown("**Multimodal Sarcasm Detection**")
st.sidebar.markdown("*Using Text & Speech Prosody*")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📝 Text Analysis",
    "🎙️ Audio Analysis",
    "🔗 Multimodal Analysis",
    "📊 Model Comparison"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** MUStARD (ACL 2019)")
st.sidebar.markdown("**Models:** DistilBERT, SVM, MLP Fusion")
st.sidebar.caption("Built for academic research")

zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nlp project.zip")
if os.path.exists(zip_path):
    with open(zip_path, "rb") as fp:
        st.sidebar.download_button(
            label="📦 Download NLP Project (.zip)",
            data=fp,
            file_name="nlp project.zip",
            mime="application/zip",
            use_container_width=True
        )

# Load models
models = load_all_models()
eval_results = load_evaluation_results()

# ─── Pages ───
if page == "🏠 Overview":
    st.title("🎭 SarcasTone")
    st.subheader("Multimodal Sarcasm Detection Using Text and Speech Prosody")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📖 About")
        st.markdown("""
        **SarcasTone** detects sarcasm by analyzing both what is said (text)
        and how it is said (speech prosody). It combines:

        - **Text Analysis** — DistilBERT transformer for semantic understanding
        - **Audio Analysis** — Prosodic features (pitch, energy, pauses) for tone
        - **Multimodal Fusion** — Combined text + audio for best performance

        Trained and evaluated on a multimodal sarcasm dataset containing
        690 annotated utterances with matched audio features.
        """)

    with col2:
        st.markdown("### 📊 Dataset Statistics")
        try:
            df = load_dataset_info()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Samples", len(df))
            c2.metric("Sarcastic", int(df['sarcasm'].sum()))
            c3.metric("Non-Sarcastic", int((1 - df['sarcasm']).sum()))

            show_counts = df['show'].value_counts()
            fig = px.pie(values=show_counts.values, names=show_counts.index,
                         title="Samples by TV Show")
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Dataset not loaded: {e}")

    st.markdown("---")
    st.markdown("### 🏗️ System Architecture")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📝 Text Model")
        st.markdown("DistilBERT fine-tuned for sarcasm classification")
        st.markdown("- Input: Utterance text")
        st.markdown("- Output: Sarcasm prediction + confidence")
    with col2:
        st.markdown("#### 🎙️ Audio Model")
        st.markdown("SVM on prosodic features (MFCC, mel, spectral)")
        st.markdown("- Input: Audio features")
        st.markdown("- Output: Sarcasm prediction + confidence")
    with col3:
        st.markdown("#### 🔗 Multimodal Model")
        st.markdown("MLP on fused text embeddings + audio features")
        st.markdown("- Input: Text + Audio")
        st.markdown("- Output: Combined prediction + confidence")

    # Model status
    st.markdown("---")
    st.markdown("### ✅ Model Status")
    c1, c2, c3 = st.columns(3)
    c1.success("Text Model ✓" if 'text' in models else "Text Model ✗ — Run training first")
    c2.success("Audio Model ✓" if 'audio' in models else "Audio Model ✗ — Run training first")
    c3.success("Multimodal Model ✓" if 'multimodal' in models else "Multimodal Model ✗ — Run training first")


elif page == "📝 Text Analysis":
    st.title("📝 Text-Based Sarcasm Detection")
    st.markdown("Analyze text for sarcasm using the DistilBERT model and sentiment analysis.")
    st.markdown("---")

    manifest = get_sample_manifest()
    if manifest:
        example_items = [f"{s['title']} [{s['category']}]" for s in manifest]
        example_texts = [s['text'] for s in manifest]
    else:
        example_items = [
            "1. Sarcastic: Corporate Meeting (Deadpan)",
            "2. Sarcastic: Standstill Traffic (Exaggerated)",
            "3. Sarcastic: Sheldon Style (Mocking)",
            "4. Sarcastic: False Politeness (Helpfulness)",
            "5. Sarcastic: Rainy Day (Bitter Irony)",
            "6. Sincere: Genuine Gratitude (Warm)",
            "7. Sincere: Sunny Weather (Delight)",
            "8. Sincere: Formal Notice (Neutral)",
            "9. Sincere: Congratulations (Enthusiastic)",
            "10. Sincere: Reminder Notice (Calm)"
        ]
        example_texts = [
            "Oh great, another meeting that could have been an email.",
            "I just love sitting in standstill traffic for two hours.",
            "It's just a privilege to watch your mind at work.",
            "Thank you so much for your incredibly helpful and timely advice.",
            "What a truly magnificent day to be stuck indoors working all weekend.",
            "I really appreciate all your help and dedication on this project.",
            "The weather outside is lovely and warm, perfect for a long walk.",
            "The scheduled team meeting will begin at three o clock in the conference room.",
            "Congratulations on your new promotion, you worked very hard for it!",
            "Please remember to submit your weekly progress summary before Friday afternoon."
        ]

    if "user_text" not in st.session_state:
        st.session_state.user_text = ""

    def set_example_text(t):
        st.session_state.user_text = t

    st.markdown("### 💡 Select from 10 Curated Examples")
    selected_option = st.selectbox(
        "Choose an example quote (5 Sarcastic / 5 Sincere):",
        ["-- Select an example quote --"] + example_items,
        index=0,
        key="text_example_dropdown"
    )

    if selected_option != "-- Select an example quote --":
        idx = example_items.index(selected_option)
        st.session_state.user_text = example_texts[idx]

    st.markdown("**Quick Preset Buttons:**")
    btn_cols = st.columns(5)
    for i in range(min(10, len(example_texts))):
        col_idx = i % 5
        btn_label = f"Sarcastic {i+1}" if i < 5 else f"Sincere {i-4}"
        btn_cols[col_idx].button(
            btn_label,
            key=f"text_btn_{i}",
            help=example_texts[i],
            on_click=set_example_text,
            args=(example_texts[i],),
            use_container_width=True
        )

    text_input = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.user_text,
        height=100,
        placeholder="Type or paste text here, or select an example above..."
    )

    if text_input:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🤖 Model Prediction")
            if 'text' in models:
                pred, conf = predict_text(text_input, models)
                label = LABEL_MAP[pred]

                if pred == 1:
                    st.error(f"**{label}** — Sarcasm detected!")
                else:
                    st.success(f"**{label}** — No sarcasm detected.")

                st.plotly_chart(create_gauge_chart(conf, "Prediction Confidence"),
                               use_container_width=True)
            else:
                st.warning("Text model not loaded. Please run training first.")

        with col2:
            st.markdown("### 💬 Sentiment Analysis")
            sentiment = analyze_text_sentiment(text_input)

            st.markdown(f"**Sentiment:** {sentiment['label']}")
            st.markdown(f"**Polarity:** {sentiment['polarity']:.3f} "
                        f"({'Positive' if sentiment['polarity'] > 0 else 'Negative' if sentiment['polarity'] < 0 else 'Neutral'})")
            st.markdown(f"**Subjectivity:** {sentiment['subjectivity']:.3f}")

            # Polarity bar
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment['polarity'],
                title={'text': 'Text Polarity'},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': 'darkblue'},
                    'steps': [
                        {'range': [-1, -0.1], 'color': 'lightcoral'},
                        {'range': [-0.1, 0.1], 'color': 'lightyellow'},
                        {'range': [0.1, 1], 'color': 'lightgreen'},
                    ]
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)


elif page == "🎙️ Audio Analysis":
    st.title("🎙️ Audio-Based Sarcasm Detection")
    st.markdown("Analyze speech prosody, pitch modulation, and acoustic tone for sarcasm detection.")
    st.markdown("---")

    manifest = get_sample_manifest()
    sample_options = ["-- None / Upload My Own File --"] + [f"{s['title']} [{s['category']}]" for s in manifest]

    col_opt1, col_opt2 = st.columns([3, 2])
    with col_opt1:
        st.markdown("### 🎧 Option A: Choose from 10 Pre-Recorded Audio Samples")
        selected_audio_choice = st.selectbox(
            "Select an audio clip (5 Sarcastic / 5 Sincere):",
            sample_options,
            index=0,
            key="audio_example_dropdown"
        )
    with col_opt2:
        st.markdown("### 📤 Option B: Upload Your Own Audio")
        uploaded_file = st.file_uploader("Upload audio file (WAV, MP3, M4A):", type=['wav', 'mp3', 'm4a'], key="audio_custom_upload")

    active_audio_path = None
    default_text = ""
    is_temp = False

    if selected_audio_choice != "-- None / Upload My Own File --":
        idx = sample_options.index(selected_audio_choice) - 1
        s = manifest[idx]
        active_audio_path = os.path.join(PROJECT_ROOT, s['wav_path'])
        default_text = s['text']
        st.info(f"**Loaded Sample:** {s['title']} | **Ground Truth:** {s['label']} ({s['category']})")
        st.audio(active_audio_path)
    elif uploaded_file:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            active_audio_path = tmp.name
            is_temp = True
        st.audio(uploaded_file, format=f"audio/{suffix.strip('.')}")
        with st.spinner("Transcribing audio..."):
            default_text = transcribe_audio(active_audio_path)

    if active_audio_path:
        st.markdown("---")
        transcript = st.text_area("Transcript (editable):", value=default_text, height=80, key="audio_transcript_field")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔊 Prosody Analysis")
            try:
                from sarcastone.features.prosody_analyzer import extract_prosody_features, get_prosody_summary
                with st.spinner("Analyzing prosody..."):
                    prosody = extract_prosody_features(active_audio_path)

                summary = get_prosody_summary(prosody)
                for category, metrics in summary.items():
                    with st.expander(f"📌 {category}", expanded=True):
                        for metric_name, value in metrics.items():
                            st.markdown(f"**{metric_name}:** {value}")

                # Prosody visualization
                st.markdown("### 📈 Prosody Visualization")
                import librosa
                y, sr_val = librosa.load(active_audio_path, sr=22050)

                # Waveform
                fig_wave = go.Figure()
                time_axis = np.arange(len(y)) / sr_val
                step = max(1, len(y) // 5000)
                fig_wave.add_trace(go.Scatter(
                    x=time_axis[::step], y=y[::step],
                    mode='lines', name='Waveform', line=dict(color='steelblue', width=0.5)
                ))
                fig_wave.update_layout(title="Waveform", xaxis_title="Time (s)",
                                       yaxis_title="Amplitude", height=250,
                                       margin=dict(l=20, r=20, t=40, b=30))
                st.plotly_chart(fig_wave, use_container_width=True)

                # F0 contour
                f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                        fmax=librosa.note_to_hz('C7'), sr=sr_val)
                if f0 is not None:
                    f0_times = librosa.times_like(f0, sr=sr_val)
                    fig_f0 = go.Figure()
                    fig_f0.add_trace(go.Scatter(
                        x=f0_times, y=f0, mode='lines',
                        name='F0', line=dict(color='darkorange')
                    ))
                    fig_f0.update_layout(title="Fundamental Frequency (F0)",
                                         xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
                                         height=250, margin=dict(l=20, r=20, t=40, b=30))
                    st.plotly_chart(fig_f0, use_container_width=True)

                # RMS Energy
                rms = librosa.feature.rms(y=y)[0]
                rms_times = librosa.times_like(rms, sr=sr_val)
                fig_rms = go.Figure()
                fig_rms.add_trace(go.Scatter(
                    x=rms_times, y=rms, mode='lines',
                    name='RMS Energy', line=dict(color='crimson'), fill='tozeroy'
                ))
                fig_rms.update_layout(title="RMS Energy (Voice Intensity)",
                                       xaxis_title="Time (s)", yaxis_title="Energy",
                                       height=250, margin=dict(l=20, r=20, t=40, b=30))
                st.plotly_chart(fig_rms, use_container_width=True)

            except Exception as e:
                st.error(f"Error analyzing prosody: {e}")

        with col2:
            st.markdown("### 🤖 Audio Model Prediction")
            if 'audio' in models:
                pred, conf = predict_audio_file(active_audio_path, models)
                label = LABEL_MAP[pred]

                if pred == 1:
                    st.error(f"**{label}** — Sarcasm detected in speech!")
                else:
                    st.success(f"**{label}** — No sarcasm detected in speech.")

                st.plotly_chart(create_gauge_chart(conf, "Audio Confidence"),
                               use_container_width=True)
            else:
                st.warning("Audio model not loaded. Please run training first.")

            # Sentiment-Tone Mismatch
            st.markdown("### 🔀 Sentiment-Tone Mismatch")
            try:
                text_sent = analyze_text_sentiment(transcript)
                acoustic = analyze_acoustic_tone(prosody)
                mismatch = detect_sentiment_tone_mismatch(text_sent, acoustic)

                st.markdown(f"**Text Sentiment:** {mismatch['text_sentiment']} "
                            f"(polarity: {mismatch['text_polarity']:.2f})")
                st.markdown(f"**Acoustic Tone:** {mismatch['acoustic_tone']} "
                            f"(score: {mismatch['tone_score']:.2f})")
                st.markdown(f"**Mismatch Level:** {mismatch['mismatch_level']}")
                st.markdown(f"**Mismatch Type:** {mismatch['mismatch_type']}")

                if mismatch['sarcasm_indicator']:
                    st.warning("⚠️ Sentiment-tone mismatch suggests possible sarcasm")
                else:
                    st.info("✓ Text and tone appear consistent")

                st.plotly_chart(create_gauge_chart(
                    mismatch['mismatch_score'], "Mismatch Score",
                    [[0, 0.3, "lightgreen"], [0.3, 0.6, "orange"], [0.6, 1.0, "red"]]
                ), use_container_width=True)
            except Exception as e:
                st.error(f"Error in mismatch analysis: {e}")

        if is_temp:
            try:
                os.unlink(active_audio_path)
            except Exception:
                pass


elif page == "🔗 Multimodal Analysis":
    st.title("🔗 Multimodal Sarcasm Detection")
    st.markdown("Combine text meaning and speech prosody for comprehensive multimodal sarcasm classification.")
    st.markdown("---")

    manifest = get_sample_manifest()
    sample_options = ["-- None / Upload My Own File --"] + [f"{s['title']} [{s['category']}]" for s in manifest]

    col_opt1, col_opt2 = st.columns([3, 2])
    with col_opt1:
        st.markdown("### 🎬 Option A: Choose from 10 Multimodal Sample Pairs")
        selected_mm_choice = st.selectbox(
            "Select a pre-paired audio + text clip (5 Sarcastic / 5 Sincere):",
            sample_options,
            index=0,
            key="mm_example_dropdown"
        )
    with col_opt2:
        st.markdown("### 📤 Option B: Upload Custom Audio")
        uploaded_file = st.file_uploader("Upload audio file (WAV, MP3, M4A):", type=['wav', 'mp3', 'm4a'], key="mm_upload")

    active_audio_path = None
    initial_transcript = ""
    is_temp = False

    if selected_mm_choice != "-- None / Upload My Own File --":
        idx = sample_options.index(selected_mm_choice) - 1
        s = manifest[idx]
        active_audio_path = os.path.join(PROJECT_ROOT, s['wav_path'])
        initial_transcript = s['text']
        st.info(f"**Loaded Multimodal Pair:** {s['title']} | **Ground Truth:** {s['label']} ({s['category']})")
        st.audio(active_audio_path)
    elif uploaded_file:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            active_audio_path = tmp.name
            is_temp = True
        st.audio(uploaded_file, format=f"audio/{suffix.strip('.')}")
        with st.spinner("Transcribing..."):
            initial_transcript = transcribe_audio(active_audio_path)

    if active_audio_path:
        transcript = st.text_area("Transcript (editable for analysis):", value=initial_transcript, height=80, key="mm_transcript")

        if st.button("🔍 Run Full Multimodal Analysis", type="primary"):
            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### 📝 Text-Only")
                if 'text' in models:
                    pred, conf = predict_text(transcript, models)
                    label = LABEL_MAP[pred]
                    if pred == 1:
                        st.error(f"**{label}**")
                    else:
                        st.success(f"**{label}**")
                    st.plotly_chart(create_gauge_chart(conf, "Text Confidence"),
                                   use_container_width=True)
                else:
                    st.warning("Text model not available")

            with col2:
                st.markdown("### 🎙️ Audio-Only")
                if 'audio' in models:
                    pred, conf = predict_audio_file(active_audio_path, models)
                    label = LABEL_MAP[pred]
                    if pred == 1:
                        st.error(f"**{label}**")
                    else:
                        st.success(f"**{label}**")
                    st.plotly_chart(create_gauge_chart(conf, "Audio Confidence"),
                                   use_container_width=True)
                else:
                    st.warning("Audio model not available")

            with col3:
                st.markdown("### 🔗 Multimodal")
                if 'multimodal' in models and 'text' in models:
                    pred, conf = predict_multimodal_input(transcript, active_audio_path, models)
                    label = LABEL_MAP[pred]
                    if pred == 1:
                        st.error(f"**{label}**")
                    else:
                        st.success(f"**{label}**")
                    st.plotly_chart(create_gauge_chart(conf, "Multimodal Confidence"),
                                   use_container_width=True)
                else:
                    st.warning("Multimodal model not available")

            # Mismatch analysis
            st.markdown("---")
            st.markdown("### 🔀 Sentiment-Tone Analysis")
            try:
                from sarcastone.features.prosody_analyzer import extract_prosody_features
                prosody = extract_prosody_features(active_audio_path)
                text_sent = analyze_text_sentiment(transcript)
                acoustic = analyze_acoustic_tone(prosody)
                mismatch = detect_sentiment_tone_mismatch(text_sent, acoustic)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Text Sentiment", mismatch['text_sentiment'])
                c2.metric("Acoustic Tone", mismatch['acoustic_tone'])
                c3.metric("Mismatch Level", mismatch['mismatch_level'])
                c4.metric("Sarcasm Indicator", "Yes ⚠️" if mismatch['sarcasm_indicator'] else "No ✓")
            except Exception as e:
                st.error(f"Error: {e}")

        if is_temp:
            try:
                os.unlink(active_audio_path)
            except Exception:
                pass


elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison Dashboard")
    st.markdown("Evaluation metrics from stratified 5-fold cross-validation.")
    st.markdown("---")

    if not eval_results:
        st.warning("No evaluation results found. Please run training first:\n"
                   "```\npython -m sarcastone.train_all\n```")
    else:
        # Summary metrics
        st.markdown("### 📋 Performance Summary")

        model_names = []
        accuracies = []
        f1_scores = []
        acc_stds = []
        f1_stds = []

        for name, display in [('text_model', 'Text (DistilBERT)'),
                               ('audio_model', 'Audio (SVM)'),
                               ('multimodal_model', 'Multimodal (Fusion)')]:
            if name in eval_results:
                r = eval_results[name]
                model_names.append(display)
                accuracies.append(r['accuracy']['mean'])
                acc_stds.append(r['accuracy']['std'])
                f1_scores.append(r['f1']['mean'])
                f1_stds.append(r['f1']['std'])

        if model_names:
            cols = st.columns(len(model_names))
            for i, (name, acc, acc_sd, f1, f1_sd) in enumerate(
                    zip(model_names, accuracies, acc_stds, f1_scores, f1_stds)):
                with cols[i]:
                    st.markdown(f"#### {name}")
                    st.metric("Accuracy", f"{acc:.4f}", f"± {acc_sd:.4f}")
                    st.metric("F1-Score", f"{f1:.4f}", f"± {f1_sd:.4f}")

            # Bar chart comparison
            st.markdown("---")
            st.markdown("### 📊 Performance Comparison")

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Accuracy', x=model_names, y=accuracies,
                                  error_y=dict(type='data', array=acc_stds),
                                  marker_color='steelblue'))
            fig.add_trace(go.Bar(name='F1-Score', x=model_names, y=f1_scores,
                                  error_y=dict(type='data', array=f1_stds),
                                  marker_color='darkorange'))
            fig.update_layout(barmode='group', height=400,
                              yaxis_title="Score", title="Model Performance Comparison",
                              yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # Per-fold results
            st.markdown("---")
            st.markdown("### 📈 Per-Fold Results")

            for name, display in [('text_model', 'Text (DistilBERT)'),
                                   ('audio_model', 'Audio (SVM)'),
                                   ('multimodal_model', 'Multimodal (Fusion)')]:
                if name in eval_results and 'accuracy' in eval_results[name]:
                    with st.expander(f"📌 {display} — Fold Details"):
                        r = eval_results[name]
                        fold_df = pd.DataFrame({
                            'Fold': [f"Fold {i+1}" for i in range(len(r['accuracy']['per_fold']))],
                            'Accuracy': r['accuracy']['per_fold'],
                            'F1-Score': r['f1']['per_fold']
                        })
                        st.dataframe(fold_df, use_container_width=True)

                        fig_fold = go.Figure()
                        fig_fold.add_trace(go.Scatter(
                            x=fold_df['Fold'], y=fold_df['Accuracy'],
                            mode='lines+markers', name='Accuracy'
                        ))
                        fig_fold.add_trace(go.Scatter(
                            x=fold_df['Fold'], y=fold_df['F1-Score'],
                            mode='lines+markers', name='F1-Score'
                        ))
                        fig_fold.update_layout(height=300, yaxis_range=[0, 1])
                        st.plotly_chart(fig_fold, use_container_width=True)

            # Confusion matrices
            st.markdown("---")
            st.markdown("### 🔢 Confusion Matrices")

            cm_cols = st.columns(3)
            for i, (name, display) in enumerate([('text_model', 'Text'),
                                                   ('audio_model', 'Audio'),
                                                   ('multimodal_model', 'Multimodal')]):
                if name in eval_results and 'confusion_matrix_sum' in eval_results[name]:
                    with cm_cols[i]:
                        cm = np.array(eval_results[name]['confusion_matrix_sum'])
                        fig_cm = create_confusion_matrix_fig(cm, f"{display} (Summed)")
                        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("### ℹ️ Methodology")
    st.markdown("""
    - **Evaluation:** Stratified 5-fold cross-validation ensures balanced class distribution in each fold.
    - **No data leakage:** Models are trained and evaluated on strictly separate folds.
    - **Metrics reported:** Mean ± standard deviation across all 5 folds.
    - **Dataset:** 690 utterances (345 sarcastic, 345 non-sarcastic).
    """)

    st.markdown("---")
    st.caption("SarcasTone — Multimodal Sarcasm Detection System.")
