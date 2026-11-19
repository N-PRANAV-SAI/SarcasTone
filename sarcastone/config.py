import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Data files
SARCASM_DATA_PATH = os.path.join(DATA_DIR, "sarcasm_data.json")
AUDIO_FEATURES_PATH = os.path.join(DATA_DIR, "audio_features.p")
SPLIT_INDICES_PATH = os.path.join(DATA_DIR, "split_indices.p")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model hyperparameters
TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_MAX_LENGTH = 128
TEXT_BATCH_SIZE = 16
TEXT_LEARNING_RATE = 2e-5
TEXT_EPOCHS = 4

# Audio features
AUDIO_N_MFCC = 13
AUDIO_SR = 22050
AUDIO_HOP_LENGTH = 512

# Cross-validation
N_FOLDS = 5
RANDOM_STATE = 42

# Labels
LABEL_MAP = {0: "Not Sarcastic", 1: "Sarcastic"}
