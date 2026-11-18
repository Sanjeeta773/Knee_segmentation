
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from args import (
    TRAIN_CSV,
    VAL_CSV,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DEVICE,
    BEST_MODEL_PATH,
    RANDOM_SEED,
)

from dataset import KneeDataset
from model import UNet
from utils import dice_loss_from_logits, dice_score_from_logits


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        bce_loss = criterion(logits, masks)
        dice_loss = dice_loss_from_logits(logits, masks)

        loss = bce_loss + dice_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)

        bce_loss = criterion(logits, masks)
        dice_loss = dice_loss_from_logits(logits, masks)

        loss = bce_loss + dice_loss

        dice_score = dice_score_from_logits(logits, masks)

        total_loss += loss.item()
        total_dice += dice_score

    average_loss = total_loss / len(loader)
    average_dice = total_dice / len(loader)

    return average_loss, average_dice


def train_model():
    torch.manual_seed(RANDOM_SEED)

    print("Device:", DEVICE)

    # Load training and validation datasets
    train_dataset = KneeDataset(TRAIN_CSV)
    val_dataset = KneeDataset(VAL_CSV)

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Create U-Net model
    model = UNet(
        n_inputs=1,
        n_classes=1
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_dice = -1.0

    # Training loop
    for epoch in range(EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_dice = validate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: {val_loss:.4f} "
            f"Validation Dice: {val_dice:.4f}"
        )

        # Save the model when validation Dice improves
        if val_dice > best_dice:
            best_dice = val_dice

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_dice": best_dice,
                },
                BEST_MODEL_PATH
            )

            print("Best model saved!")

    print("\nTraining completed!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    print("Best model location:", BEST_MODEL_PATH)


if __name__ == "__main__":
    train_model()
