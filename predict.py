
import os
import sys
import joblib
import pandas as pd

# Configure UTF-8 streams on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from config import MODEL_PATH


def main():
    # Check whether training has been completed
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run python train.py first.")
        return

    # Load the trained model
    model = joblib.load(MODEL_PATH)

    print("Student Marks Prediction")
    print("------------------------")

    while True:
        try:
            user_input = input(
                "Enter study hours (or type 'q' to quit): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nProgram stopped.")
            break

        if user_input.lower() == "q":
            print("Program stopped.")
            break

        try:
            hours = float(user_input)

            if not 0 <= hours <= 24:
                print("Enter a value between 0 and 24.")
                continue

            # Use the same feature name as in training
            sample = pd.DataFrame({"hours": [hours]})

            predicted_marks = model.predict(sample)[0]

            # Clip only for display to a marks scale of 0–100
            display_marks = max(0, min(100, predicted_marks))

            print(f"Predicted marks: {display_marks:.2f}/100\n")

        except ValueError:
            print("Please enter a valid number or 'q'.")


if __name__ == "__main__":
    main()
