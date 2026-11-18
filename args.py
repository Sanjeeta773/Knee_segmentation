
from pathlib import Path
import torch


# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset paths
DATA_DIR = BASE_DIR / "data"
CSV_DIR = DATA_DIR / "CSVs"

TRAIN_CSV = CSV_DIR / "train.csv"
VAL_CSV = CSV_DIR / "validation.csv"
TEST_CSV = CSV_DIR / "test.csv"

# Model settings
IMAGE_SIZE = 256 
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1

# Training settings
BATCH_SIZE = 1
EPOCHS = 30
LEARNING_RATE = 0.0001

# Device settings
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Output directories
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# Reproducibility
RANDOM_SEED = 42
