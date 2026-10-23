from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix


PROJECT_DIR = Path(
    r"C:\Users\AlHuda\Desktop\AI WorkSpace\work-base Projects\S1 L3\FitOrbit"
)

TEST_DATA_PATH = PROJECT_DIR / "preprocessed_test.npz"
MODEL_PATH = PROJECT_DIR / "fit_orbit_bilstm.keras"
ENCODER_PATH = PROJECT_DIR / "label_encoder.pkl"

RESULTS_DIR = PROJECT_DIR / "training_results"


def main():
    for path in (TEST_DATA_PATH, MODEL_PATH, ENCODER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"الملف غير موجود:\n{path}")

    # تحميل بيانات الاختبار والموديل والـEncoder المحفوظ
    test_data = np.load(TEST_DATA_PATH, allow_pickle=False)
    X_test = test_data["X_test"].astype(np.float32)
    y_test = test_data["y_test"].astype(np.int64)

    encoder = joblib.load(ENCODER_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    if len(X_test) == 0:
        raise ValueError("ملف الاختبار لا يحتوي على أي تسلسلات.")

    if len(X_test) != len(y_test):
        raise ValueError(
            f"عدد التسلسلات ({len(X_test)}) لا يساوي عدد التصنيفات ({len(y_test)})."
        )

    # تقييم شامل
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    probabilities = model.predict(X_test, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    class_ids = np.arange(len(encoder.classes_))
    class_names = [str(name) for name in encoder.classes_]

    report = classification_report(
        y_test,
        y_pred,
        labels=class_ids,
        target_names=class_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=class_ids,
    )

    print("نتائج الاختبار الخارجي")
    print(f"عدد التسلسلات: {len(X_test)}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix:")
    print(matrix)

    # حفظ النتائج
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    ).to_csv(RESULTS_DIR / "external_test_confusion_matrix.csv")

    with open(
        RESULTS_DIR / "external_test_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    # حفظ التوقع والتصنيف الصحيح لكل تسلسل
    predictions_df = pd.DataFrame({
        "true_label": encoder.inverse_transform(y_test),
        "predicted_label": encoder.inverse_transform(y_pred),
        "correct": y_test == y_pred,
    })

    if "test_video_ids" in test_data.files:
        predictions_df["video_id"] = test_data["test_video_ids"].astype(str)

    if "test_start_frames" in test_data.files:
        predictions_df["start_frame"] = test_data["test_start_frames"]

    predictions_df.to_csv(
        RESULTS_DIR / "external_test_predictions.csv",
        index=False,
    )

    print(f"\nتم حفظ النتائج في:\n{RESULTS_DIR}")


if __name__ == "__main__":
    main()