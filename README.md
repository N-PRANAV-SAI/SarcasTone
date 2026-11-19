# SarcasTone – Multimodal Sarcasm Detection Using Text and Speech Prosody

A multimodal sarcasm detection system that combines textual meaning and speech prosody analysis.

## Features

- **Text-Only Model**: Fine-tuned DistilBERT for sarcasm classification
- **Audio-Only Model**: SVM classifier on prosodic features (MFCC, mel-spectrogram, spectral centroid)
- **Multimodal Model**: MLP on fused DistilBERT embeddings + audio features
- **Prosody Analysis**: F0, pitch variation, RMS energy, speaking rate, pause detection
- **Sentiment-Tone Mismatch**: Compares text sentiment with acoustic tone
- **Streamlit Dashboard**: Interactive web interface with visualizations

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Training

Train all three models with 5-fold stratified cross-validation:

```bash
python -m sarcastone.train_all
```

This will:
1. Train the text model (DistilBERT) — ~10 min
2. Train the audio model (SVM) — ~1 min
3. Train the multimodal model (MLP fusion) — ~5 min
4. Save models to `models/` and results to `results/`

## Running the App

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                      # Streamlit application
├── requirements.txt            # Dependencies
├── sarcastone/
│   ├── config.py               # Configuration
│   ├── data_utils.py           # Data loading & splitting
│   ├── evaluate.py             # Evaluation metrics
│   ├── train_all.py            # Master training script
│   ├── features/
│   │   ├── prosody_analyzer.py # Prosody feature extraction
│   │   └── sentiment_analyzer.py # Sentiment-tone mismatch
│   └── models/
│       ├── text_model.py       # DistilBERT text classifier
│       ├── audio_model.py      # SVM audio classifier
│       └── multimodal_model.py # Fused text+audio classifier
├── data/
│   ├── sarcasm_data.json       # Dataset (690 utterances)
│   ├── audio_features.p        # Pre-extracted audio features
│   └── split_indices.p         # 5-fold CV indices
├── models/                     # Saved trained models
└── results/                    # Evaluation results
```

## Evaluation

All models are evaluated with stratified 5-fold cross-validation. Metrics include accuracy, precision, recall, F1-score, and confusion matrices.

## License

MIT License — see [LICENSE](LICENSE) for details.
