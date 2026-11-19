import numpy as np
from typing import Dict, Any, Optional


def analyze_text_sentiment(text):
    """Analyze text sentiment using TextBlob.

    Returns dict with polarity (-1 to 1), subjectivity (0 to 1), and label.
    """
    try:
        from textblob import TextBlob
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.1:
            label = 'Positive'
        elif polarity < -0.1:
            label = 'Negative'
        else:
            label = 'Neutral'

        return {
            'polarity': float(polarity),
            'subjectivity': float(subjectivity),
            'label': label
        }
    except ImportError:
        return {'polarity': 0.0, 'subjectivity': 0.0, 'label': 'Unknown'}


def analyze_acoustic_tone(prosody_features):
    """Characterize the acoustic tone from prosody features.

    Uses pitch variation, energy patterns, and speaking rate to infer
    acoustic emotional characteristics.
    """
    if prosody_features is None:
        return {
            'energy_level': 'Unknown',
            'pitch_variation': 'Unknown',
            'speaking_pace': 'Unknown',
            'overall_tone': 'Unknown',
            'tone_score': 0.0
        }

    # Energy level
    rms_mean = prosody_features.get('rms_mean', 0)
    if rms_mean > 0.05:
        energy_level = 'High'
    elif rms_mean > 0.02:
        energy_level = 'Medium'
    else:
        energy_level = 'Low'

    # Pitch variation
    f0_std = prosody_features.get('f0_std', 0)
    f0_range = prosody_features.get('f0_range', 0)
    if f0_std > 50 or f0_range > 150:
        pitch_variation = 'High (exaggerated)'
    elif f0_std > 25 or f0_range > 80:
        pitch_variation = 'Moderate'
    else:
        pitch_variation = 'Low (flat/monotone)'

    # Speaking pace
    speaking_rate = prosody_features.get('speaking_rate', 0)
    if speaking_rate > 5:
        speaking_pace = 'Fast'
    elif speaking_rate > 3:
        speaking_pace = 'Normal'
    else:
        speaking_pace = 'Slow'

    # Overall tone score (-1 to 1, negative = flat/ironic, positive = enthusiastic/genuine)
    tone_score = 0.0
    if f0_std > 50:
        tone_score += 0.3
    elif f0_std < 15:
        tone_score -= 0.3

    if rms_mean > 0.05:
        tone_score += 0.2
    elif rms_mean < 0.02:
        tone_score -= 0.2

    pause_rate = prosody_features.get('pause_rate', 0)
    if pause_rate > 0.2:
        tone_score -= 0.2

    emphasis = prosody_features.get('emphasis_ratio', 1.0)
    if emphasis > 3.0:
        tone_score -= 0.1

    if tone_score > 0.1:
        overall_tone = 'Genuine/Enthusiastic'
    elif tone_score < -0.1:
        overall_tone = 'Flat/Ironic'
    else:
        overall_tone = 'Neutral'

    return {
        'energy_level': energy_level,
        'pitch_variation': pitch_variation,
        'speaking_pace': speaking_pace,
        'overall_tone': overall_tone,
        'tone_score': float(np.clip(tone_score, -1, 1))
    }


def detect_sentiment_tone_mismatch(text_sentiment, acoustic_tone):
    """Detect mismatch between text sentiment and acoustic tone.

    A mismatch (e.g., positive text + flat/ironic tone) is a sarcasm indicator.
    """
    text_polarity = text_sentiment.get('polarity', 0)
    text_label = text_sentiment.get('label', 'Neutral')
    tone_score = acoustic_tone.get('tone_score', 0)
    tone_label = acoustic_tone.get('overall_tone', 'Unknown')

    mismatch_score = 0.0

    if text_polarity > 0.1 and tone_score < -0.1:
        mismatch_score = abs(text_polarity - tone_score) / 2
        mismatch_type = 'Positive text with flat/ironic delivery'
    elif text_polarity < -0.1 and tone_score > 0.1:
        mismatch_score = abs(text_polarity - tone_score) / 2
        mismatch_type = 'Negative text with enthusiastic delivery'
    elif abs(text_polarity) > 0.3 and abs(tone_score) < 0.05:
        mismatch_score = abs(text_polarity) * 0.3
        mismatch_type = 'Strong sentiment with neutral delivery'
    else:
        mismatch_score = 0.0
        mismatch_type = 'No significant mismatch'

    mismatch_score = float(np.clip(mismatch_score, 0, 1))

    if mismatch_score > 0.3:
        mismatch_level = 'High'
    elif mismatch_score > 0.1:
        mismatch_level = 'Moderate'
    else:
        mismatch_level = 'Low'

    return {
        'text_sentiment': text_label,
        'text_polarity': float(text_polarity),
        'acoustic_tone': tone_label,
        'tone_score': float(tone_score),
        'mismatch_score': mismatch_score,
        'mismatch_level': mismatch_level,
        'mismatch_type': mismatch_type,
        'sarcasm_indicator': mismatch_score > 0.2
    }
