# Knee X-ray Tibia Segmentation Using U-Net

## 1. Project Overview

This project focuses on segmenting the tibia bone from knee X-ray images using a U-Net deep learning model.

The main goal is to train a model that can identify the tibia region in an X-ray image and produce a segmentation mask. This helps separate the bone from the surrounding structures in the image.


## 2. Dataset

The dataset contains 150 knee X-ray images:

- 100 images with corresponding segmentation masks.
- 50 additional images without masks, used for prediction.

The labelled data was divided into:

- Training set: 80 images
- Validation set: 20 images

The remaining 50 images were kept as the test set.

The images and masks are stored in separate folders. CSV files are used to organize the dataset paths.

## 3. Methodology

The project uses a U-Net architecture for image segmentation.

The main steps are:

1. Load the knee X-ray images and their corresponding masks.
2. Convert the images to grayscale and resize them to 256 × 256 pixels.
3. Normalize the image pixel values.
4. Train the U-Net model using the training data.
5. Use the validation data to monitor model performance.
6. Save the best model during training.
7. Load the saved model and generate predictions for the 50 test images.

## 4. Model

U-Net is a convolutional neural network commonly used for image segmentation.

It has an encoder and a decoder. The encoder extracts image features, while the decoder uses these features to produce a segmentation mask.

In this project, the model is designed to produce a binary mask of the tibia region.

## 5. Evaluation

The model's performance is evaluated using the Dice score, which measures the overlap between the predicted segmentation mask and the ground-truth mask.

The validation dataset contains labelled images and can be used to calculate the Dice score.

The 50 test images do not have ground-truth masks, so predictions can be generated for them, but a test Dice score cannot be calculated without reference masks.

## 6. Project Structure

```text
Knee_segmentation/
│
├── data/
│   ├── xrays/
│   ├── masks/
│   └── CSVs/
│       ├── dataset.csv
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── checkpoints/
├── outputs/
├── deeppipeline/
│
├── args.py
├── create_csvs.py
├── dataset.py
├── evaluate.py
├── main.py
├── model.py
├── trainer.py
├── utils.py
│
└── README.md
```

## 7. Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- OpenCV
- Matplotlib
- U-Net

## 8. How to Run the Project

1. Clone the repository.
2. Install the required Python packages.
3. Make sure the dataset is organized in the correct folders.
4. Run the main Python script to start training and evaluation.

The exact commands and package requirements should follow the project setup.

## 9. Conclusion

This project gave me practical experience with medical image segmentation and deep learning. I learned how to organize X-ray images and masks, work with a U-Net model, train a segmentation model, and generate predicted masks for new images.

The project focuses on identifying the tibia region in knee X-ray images.