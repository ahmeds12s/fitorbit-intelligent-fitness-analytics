from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(
    r"C:\Users\AlHuda\Desktop\AI WorkSpace\work-base Projects\S1 L3\FitOrbit"
)

INPUT_CSV = PROJECT_DIR / "paper_test.csv"
OUTPUT_NPZ = PROJECT_DIR / "preprocessed_test.npz"
SCALER_PATH = PROJECT_DIR / "scaler.pkl"
ENCODER_PATH = PROJECT_DIR / "label_encoder.pkl"

WINDOW_SIZE = 30

METADATA_COLUMNS = {
    "video_id",
    "label",
    "frame_number",
    "frame_id",
    "timestamp_ms",
}


def find_frame_column(data):
    """يدعم اسمي العمود المستخدمين في ملفاتك."""
    if "frame_number" in data.columns:
        return "frame_number"
    if "frame_id" in data.columns:
        return "frame_id"

    raise ValueError(
        "لم أجد عمود frame_number أو frame_id في ملف CSV."
    )


def create_test_sequences(data, feature_columns, frame_column):
    """
    ينشئ نوافذ اختبار غير متداخلة من 30 إطارًا متتاليًا.
    لا يضمّ الإطارات التي بينها فجوة في أرقام الإطارات.
    """
    X, y_text = [], []
    video_ids, start_frames = [], []

    for video_id, group in data.groupby("video_id", sort=False):
        group = (
            group.sort_values(frame_column)
            .drop_duplicates(frame_column, keep="first")
            .reset_index(drop=True)
        )

        labels = group["label"].dropna().astype(str).unique()

        if len(labels) != 1:
            print(f"تجاهلت {video_id}: التصنيف غير موحد.")
            continue

        frames = group[frame_column].to_numpy(dtype=np.int64)
        features = group[feature_columns].to_numpy(dtype=np.float32)

        # تقسيم الفيديو إلى مقاطع متصلة
        breaks = np.where(np.diff(frames) != 1)[0] + 1
        segments = np.split(np.arange(len(frames)), breaks)

        for segment in segments:
            if len(segment) < WINDOW_SIZE:
                continue

            segment_features = features[segment]
            segment_frames = frames[segment]

            # نوافذ غير متداخلة: 0-29، 30-59، ...
            for start in range(
                0,
                len(segment_features) - WINDOW_SIZE + 1,
                WINDOW_SIZE,
            ):
                end = start + WINDOW_SIZE

                X.append(segment_features[start:end])
                y_text.append(labels[0])
                video_ids.append(str(video_id))
                start_frames.append(int(segment_frames[start]))

    if not X:
        raise ValueError(
            f"لم ينتج أي تسلسل كامل من {WINDOW_SIZE} إطارًا. "
            "راجعي عدد الإطارات المتصلة في فيديوهات الاختبار."
        )

    return (
        np.stack(X).astype(np.float32),
        np.asarray(y_text, dtype=str),
        np.asarray(video_ids, dtype=str),
        np.asarray(start_frames, dtype=np.int64),
    )


def main():
    for path in (INPUT_CSV, SCALER_PATH, ENCODER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"الملف غير موجود:\n{path}")

    data = pd.read_csv(INPUT_CSV)

    required_columns = {"video_id", "label"}
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            "أعمدة مطلوبة غير موجودة: "
            + ", ".join(sorted(missing))
        )

    frame_column = find_frame_column(data)

    # نحافظ على ترتيب أعمدة الميزات كما هو في CSV.
    # هذا الترتيب يجب أن يطابق ترتيب ميزات التدريب.
    feature_columns = [
        column
        for column in data.columns
        if column not in METADATA_COLUMNS
    ]

    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    expected_features = getattr(scaler, "n_features_in_", None)
    if expected_features is not None and len(feature_columns) != expected_features:
        raise ValueError(
            f"ملف الاختبار يحتوي على {len(feature_columns)} ميزة، "
            f"لكن الـScaler المحفوظ يتوقع {expected_features}.\n"
            "تأكدي أن أعمدة الميزات وترتيبها يطابقان بيانات التدريب."
        )

    data["video_id"] = data["video_id"].astype(str)
    data["label"] = data["label"].astype(str)
    data[frame_column] = pd.to_numeric(
        data[frame_column],
        errors="raise",
    ).astype(np.int64)

    data[feature_columns] = data[feature_columns].apply(
        pd.to_numeric,
        errors="raise",
    )

    feature_values = data[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(feature_values).all():
        raise ValueError("توجد قيم NaN أو لا نهائية في ميزات الاختبار.")

    # إنشاء نوافذ الاختبار
    X_test, y_test_text, test_video_ids, test_start_frames = (
        create_test_sequences(data, feature_columns, frame_column)
    )

    # التأكد من أن التصنيفات موجودة في الـEncoder المحفوظ
    known_classes = set(map(str, encoder.classes_))
    unknown_labels = set(y_test_text) - known_classes

    if unknown_labels:
        raise ValueError(
            "هناك تصنيفات اختبار غير موجودة في الـLabelEncoder المحفوظ: "
            + ", ".join(sorted(unknown_labels))
        )

    # استخدام الـEncoder المحفوظ فقط، بدون fit
    y_test = encoder.transform(y_test_text)

    # استخدام الـScaler المحفوظ فقط، بدون fit
    original_shape = X_test.shape
    X_test = scaler.transform(
        X_test.reshape(-1, original_shape[-1])
    ).reshape(original_shape).astype(np.float32)

    np.savez_compressed(
        OUTPUT_NPZ,
        X_test=X_test,
        y_test=y_test,
        y_test_text=y_test_text,
        classes=encoder.classes_,
        feature_columns=np.asarray(feature_columns),
        test_video_ids=test_video_ids,
        test_start_frames=test_start_frames,
    )

    print("تم تجهيز بيانات الاختبار الخارجي.")
    print("عدد الفيديوهات:", len(np.unique(test_video_ids)))
    print("عدد التسلسلات:", len(X_test))
    print("شكل X_test:", X_test.shape)
    print("شكل y_test:", y_test.shape)
    print("الفئات:", list(encoder.classes_))
    print("تم الحفظ في:", OUTPUT_NPZ)


if __name__ == "__main__":
    main()