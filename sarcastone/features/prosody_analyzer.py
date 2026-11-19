import numpy as np


def extract_prosody_features(audio_path, sr=22050):
    """Extract comprehensive prosody features from an audio file.

    Returns a dict of prosody features with robust error handling.
    """
    import librosa

    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return get_default_prosody_features()

    if len(y) == 0:
        return get_default_prosody_features()

    features = {}

    # 1. Fundamental Frequency (F0) using pyin
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr
        )
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        if len(f0_valid) > 0:
            features['f0_mean'] = float(np.mean(f0_valid))
            features['f0_std'] = float(np.std(f0_valid))
            features['f0_min'] = float(np.min(f0_valid))
            features['f0_max'] = float(np.max(f0_valid))
            features['f0_range'] = float(np.max(f0_valid) - np.min(f0_valid))
            if len(f0_valid) > 1:
                slope = np.polyfit(np.arange(len(f0_valid)), f0_valid, 1)[0]
                features['f0_slope'] = float(slope)
            else:
                features['f0_slope'] = 0.0
            features['voiced_fraction'] = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
        else:
            for key in ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range', 'f0_slope', 'voiced_fraction']:
                features[key] = 0.0
    except Exception:
        for key in ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range', 'f0_slope', 'voiced_fraction']:
            features[key] = 0.0

    # 2. Voice Intensity (RMS Energy)
    try:
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        features['rms_max'] = float(np.max(rms))
        features['dynamic_range'] = float(np.max(rms) - np.min(rms)) if len(rms) > 0 else 0.0
    except Exception:
        for key in ['rms_mean', 'rms_std', 'rms_max', 'dynamic_range']:
            features[key] = 0.0

    # 3. Speaking Rate (onset-based estimate)
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
        duration = librosa.get_duration(y=y, sr=sr)
        features['speaking_rate'] = float(len(onsets) / duration) if duration > 0 else 0.0
        features['duration'] = float(duration)
        features['num_onsets'] = len(onsets)
    except Exception:
        features['speaking_rate'] = 0.0
        features['duration'] = 0.0
        features['num_onsets'] = 0

    # 4. Silence and Pause Detection
    try:
        intervals = librosa.effects.split(y, top_db=30)
        if len(intervals) > 1:
            pauses = []
            for i in range(1, len(intervals)):
                pause_duration = (intervals[i][0] - intervals[i-1][1]) / sr
                pauses.append(pause_duration)
            features['num_pauses'] = len(pauses)
            features['mean_pause_duration'] = float(np.mean(pauses))
            features['max_pause_duration'] = float(np.max(pauses))
            features['total_pause_duration'] = float(np.sum(pauses))
            duration = librosa.get_duration(y=y, sr=sr)
            features['pause_rate'] = float(features['total_pause_duration'] / duration) if duration > 0 else 0.0
        else:
            for key in ['num_pauses', 'mean_pause_duration', 'max_pause_duration', 'total_pause_duration', 'pause_rate']:
                features[key] = 0.0
    except Exception:
        for key in ['num_pauses', 'mean_pause_duration', 'max_pause_duration', 'total_pause_duration', 'pause_rate']:
            features[key] = 0.0

    # 5. Spectral Features
    try:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
    except Exception:
        for key in ['spectral_centroid_mean', 'spectral_centroid_std', 'spectral_rolloff_mean',
                     'spectral_bandwidth_mean', 'zcr_mean', 'zcr_std']:
            features[key] = 0.0

    # 6. MFCCs
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
    except Exception:
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = 0.0
            features[f'mfcc_{i+1}_std'] = 0.0

    # 7. Energy-based emphasis/stress
    try:
        rms = librosa.feature.rms(y=y)[0]
        if len(rms) > 0:
            features['emphasis_ratio'] = float(np.max(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0.0
            features['energy_cv'] = float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0.0
        else:
            features['emphasis_ratio'] = 0.0
            features['energy_cv'] = 0.0
    except Exception:
        features['emphasis_ratio'] = 0.0
        features['energy_cv'] = 0.0

    return features


def get_default_prosody_features():
    """Return default prosody features with zero values."""
    keys = [
        'f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range', 'f0_slope', 'voiced_fraction',
        'rms_mean', 'rms_std', 'rms_max', 'dynamic_range',
        'speaking_rate', 'duration', 'num_onsets',
        'num_pauses', 'mean_pause_duration', 'max_pause_duration', 'total_pause_duration', 'pause_rate',
        'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_rolloff_mean',
        'spectral_bandwidth_mean', 'zcr_mean', 'zcr_std',
        'emphasis_ratio', 'energy_cv'
    ]
    for i in range(13):
        keys.append(f'mfcc_{i+1}_mean')
        keys.append(f'mfcc_{i+1}_std')
    return {k: 0.0 for k in keys}


def prosody_to_vector(features_dict):
    """Convert prosody features dict to a fixed-length numpy vector."""
    default = get_default_prosody_features()
    keys = sorted(default.keys())
    return np.array([float(features_dict.get(k, 0.0)) for k in keys])


def get_prosody_summary(features_dict):
    """Return a human-readable summary of prosody features."""
    summary = {}

    summary['Fundamental Frequency (F0)'] = {
        'Mean': f"{features_dict.get('f0_mean', 0):.1f} Hz",
        'Std Dev': f"{features_dict.get('f0_std', 0):.1f} Hz",
        'Range': f"{features_dict.get('f0_range', 0):.1f} Hz",
        'Trend': 'Rising' if features_dict.get('f0_slope', 0) > 0 else 'Falling'
    }

    summary['Voice Intensity'] = {
        'Mean RMS': f"{features_dict.get('rms_mean', 0):.4f}",
        'Dynamic Range': f"{features_dict.get('dynamic_range', 0):.4f}",
        'Emphasis Ratio': f"{features_dict.get('emphasis_ratio', 0):.2f}x"
    }

    summary['Speaking Rate'] = {
        'Onsets/sec': f"{features_dict.get('speaking_rate', 0):.1f}",
        'Duration': f"{features_dict.get('duration', 0):.2f}s"
    }

    summary['Pauses'] = {
        'Count': f"{int(features_dict.get('num_pauses', 0))}",
        'Mean Duration': f"{features_dict.get('mean_pause_duration', 0):.3f}s",
        'Pause Rate': f"{features_dict.get('pause_rate', 0)*100:.1f}%"
    }

    return summary


def extract_audio_features_for_model(audio_path, sr=22050):
    """Extract features suitable for the audio classifier from a new audio file.

    Returns a 283-dim vector matching the pre-extracted MUStARD features format:
    13 MFCC + 13 MFCC delta + 128 mel + 128 mel delta + 1 spectral centroid
    All mean-pooled across time.
    """
    import librosa

    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return np.zeros(283)

    if len(y) == 0:
        return np.zeros(283)

    hop_length = 512

    try:
        # Vocal isolation (same as original MUStARD)
        D = librosa.stft(y, hop_length=hop_length)
        S_full, phase = librosa.magphase(D)
        S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine',
                                                width=int(librosa.time_to_frames(0.2, sr=sr)))
        S_filter = np.minimum(S_full, S_filter)
        margin_v = 4
        power = 2
        mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
        S_foreground = mask_v * S_full
        new_D = S_foreground * phase
        y = librosa.istft(new_D)
    except Exception:
        pass  # Use original signal if vocal isolation fails

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_delta = librosa.feature.delta(S)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        features = np.concatenate([
            np.mean(mfcc, axis=1),              # 13
            np.mean(mfcc_delta, axis=1),        # 13
            np.mean(S, axis=1),                 # 128
            np.mean(S_delta, axis=1),           # 128
            np.mean(spectral_centroid, axis=1)  # 1
        ])  # Total: 283

        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(283)
