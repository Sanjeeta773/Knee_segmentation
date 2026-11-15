
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class KneeDataset(Dataset):
    def __init__(self, csv_file, image_size=256):
        self.data = pd.read_csv(csv_file)
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Load X-ray image
        image_path = Path(row["xrays"])
        image = Image.open(image_path).convert("L")
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        # Convert image to tensor and normalize
        image = torch.tensor(
            list(image.getdata()), dtype=torch.float32
        ).reshape(1, self.image_size, self.image_size) / 255.0

        # Check whether a mask exists
        mask_value = row["masks"]

        if pd.isna(mask_value) or str(mask_value).strip() == "None":
            return image, None

        # Load binary segmentation mask
        mask_path = Path(mask_value)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        mask = torch.tensor(
            list(mask.getdata()), dtype=torch.float32
        ).reshape(1, self.image_size, self.image_size)

        # Convert mask to binary values: 0 or 1
        mask = (mask > 127).float()

        return image, mask
