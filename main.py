
from trainer import train_model
from evaluate import main as evaluate_model


def main():
    print("=" * 50)
    print("KNEE X-RAY BONE SEGMENTATION")
    print("=" * 50)

    # Step 1: Train the U-Net model
    print("\nSTEP 1: MODEL TRAINING")
    train_model()

    # Step 2: Evaluate the best saved model
    print("\nSTEP 2: MODEL EVALUATION AND TEST PREDICTION")
    evaluate_model()

    print("\n" + "=" * 50)
    print("PROJECT PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    main()
