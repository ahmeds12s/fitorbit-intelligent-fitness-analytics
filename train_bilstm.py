from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Input,
    LSTM,
)


DATA_PATH = Path(r"C:\Users\AlHuda\Desktop\AI WorkSpace\work-base Projects\S1 L3\FitOrbit\preprocessed_data.npz")
MODEL_PATH = Path("fit_orbit_bilstm.keras")
OUTPUT_DIR = Path("training_results")

RANDOM_SEED = 42
BATCH_SIZE = 54
EPOCHS = 73
UNITS = 73
DROPOUT_RATE = 0.2174
LEARNING_RATE = 0.0004


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"لم أجد ملف البيانات: {DATA_PATH.resolve()}"
        )

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    data = np.load(DATA_PATH, allow_pickle=False)

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    sample_weight_train = data["sample_weight_train"].astype(np.float32)

    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    classes = data["classes"].astype(str)

    # تأكد أن أبعاد البيانات مناسبة للنموذج
    if X_train.ndim != 3:
        raise ValueError(f"شكل X_train غير متوقع: {X_train.shape}")

    if X_train.shape[1:] != (30, 78):
        raise ValueError(
            f"توقعت شكلًا (عدد العينات، 30، 78)، "
            f"لكن وجدت {X_train.shape}."
        )

    if len(X_train) != len(y_train):
        raise ValueError("عدد تسلسلات التدريب لا يطابق عدد التصنيفات.")

    if len(X_train) != len(sample_weight_train):
        raise ValueError("عدد أوزان التدريب لا يطابق عدد التسلسلات.")

    if len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("بيانات التحقق أو الاختبار الداخلي فارغة.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    number_of_classes = len(classes)

    model = Sequential([
        Input(shape=(30, 78)),

        Bidirectional(
            LSTM(UNITS, return_sequences=True)
        ),
        Dropout(DROPOUT_RATE),

        Bidirectional(
            LSTM(UNITS)
        ),
        Dropout(DROPOUT_RATE),

        Dense(number_of_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # احفظ سجل التدريب
    history_path = OUTPUT_DIR / "training_history.csv"
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    # حمّل أفضل نموذج محفوظ حسب أداء التحقق
    best_model = tf.keras.models.load_model(MODEL_PATH)

    # الاختبار الداخلي يُستخدم بعد اختيار النموذج
    test_results = best_model.evaluate(
        X_test,
        y_test,
        verbose=0,
        return_dict=True,
    )

    print("\nنتائج الاختبار الداخلي:")
    for metric_name, value in test_results.items():
        print(f"  {metric_name}: {value:.4f}")

    y_probabilities = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probabilities, axis=1)

    class_ids = np.arange(number_of_classes)

    report = classification_report(
        y_test,
        y_pred,
        labels=class_ids,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    report_path = OUTPUT_DIR / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=class_ids,
    )

    matrix_path = OUTPUT_DIR / "confusion_matrix.csv"
    pd.DataFrame(
        matrix,
        index=classes,
        columns=classes,
    ).to_csv(matrix_path)

    print("\nتقرير التصنيف:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=class_ids,
            target_names=classes,
            zero_division=0,
        )
    )

    print(f"\nأفضل نموذج: {MODEL_PATH.resolve()}")
    print(f"سجل التدريب: {history_path.resolve()}")
    print(f"تقرير التصنيف: {report_path.resolve()}")
    print(f"مصفوفة الالتباس: {matrix_path.resolve()}")

    if "X_external_test" in data.files:
        if len(data["X_external_test"]) == 0:
            print(
                "\nالاختبار الخارجي فارغ؛ "
                "my_test_video_1 لم يُستخرج بعد."
            )


if __name__ == "__main__":
    main()