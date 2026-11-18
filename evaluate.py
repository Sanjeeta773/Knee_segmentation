
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import DataLoader
from pathlib import Path

from args import (
    DEVICE,
    IMAGE_SIZE,
    BATCH_SIZE,
    VAL_CSV,
    TEST_CSV,
    BEST_MODEL_PATH,
    OUTPUT_DIR,
)

from dataset import KneeDataset
from model import UNet
from utils import dice_score_from_logits


def load_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model not found: {BEST_MODEL_PATH}. "
            "Please train the model first."
        )

    model = UNet(
        n_inputs=1,
        n_classes=1
    ).to(DEVICE)

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded best model from epoch:", checkpoint["epoch"])
    print("Best validation Dice:", checkpoint["validation_dice"])

    return model


@torch.no_grad()
def evaluate_validation(model):
    dataset = KneeDataset(
        VAL_CSV,
        image_size=IMAGE_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    scores = []

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        score = dice_score_from_logits(outputs, masks)
        scores.append(score)

    average_dice = sum(scores) / len(scores)

    print("\nValidation evaluation completed.")
    print(f"Validation Dice score: {average_dice:.4f}")

    return average_dice


@torch.no_grad()
def predict_test_images(model):
    test_data = pd.read_csv(TEST_CSV)

    prediction_dir = OUTPUT_DIR / "predicted_masks"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating predictions for test images...")

    for _, row in test_data.iterrows():
        image_path = Path(row["xrays"])

        if not image_path.is_absolute():
            image_path = Path(__file__).resolve().parent / image_path

        image = Image.open(image_path).convert("L")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        image_array = np.array(
            image,
            dtype=np.float32
        ) / 255.0

        image_tensor = torch.from_numpy(
            image_array
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        output = model(image_tensor)

        prediction = torch.sigmoid(output)
        prediction = (prediction > 0.5).float()

        mask_array = (
            prediction.squeeze().cpu().numpy() * 255
        ).astype(np.uint8)

        output_name = Path(row["xrays"]).stem + "_prediction.png"

        Image.fromarray(mask_array).save(
            prediction_dir / output_name
        )

    print("Test predictions saved in:", prediction_dir)
    print("Number of predictions:", len(test_data))


def main():
    print("Loading best U-Net model...")
    model = load_model()

    evaluate_validation(model)

    predict_test_images(model)

    print("\nEvaluation and prediction completed!")


if __name__ == "__main__":
    main()
