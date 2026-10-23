from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


INPUT_CSV = Path(r"C:\Users\AlHuda\Desktop\AI WorkSpace\work-base Projects\S1 L3\FitOrbit\paper_frame_features_v2.csv")
OUTPUT_NPZ = Path("preprocessed_data.npz")
SCALER_PATH = Path("scaler.pkl")
ENCODER_PATH = Path("label_encoder.pkl")

WINDOW_SIZE = 30
RANDOM_STATE = 42
EXTERNAL_TEST_SOURCE = "my_test_video_1"

# معلومات لا تدخل كميزات للنموذج
METADATA_COLUMNS = {
    "video_id",
    "label",
    "frame_number",
    "timestamp_ms",
}


def get_source(video_id):
    """يستخرج مصدر الفيديو من بداية مساره."""
    return str(video_id).replace("\\", "/").split("/")[0]


def create_sequences(data, selected_video_ids, feature_columns):
    """
    ينشئ نوافذ غير متداخلة من 30 إطارًا متتاليًا.
    لا يسمح للنافذة بتجاوز فجوة في frame_number.
    """
    selected_ids = set(selected_video_ids)
    subset = data[data["video_id"].isin(selected_ids)]

    X = []
    y = []
    sources = []
    video_ids = []
    start_frames = []

    for video_id, group in subset.groupby("video_id", sort=False):
        group = group.sort_values("frame_number")
        group = group.drop_duplicates("frame_number", keep="first")

        labels = group["label"].dropna().unique()
        if len(labels) != 1:
            print(f"تجاهلت {video_id}: يوجد أكثر من تصنيف.")
            continue

        label = labels[0]
        source = get_source(video_id)

        frames = group["frame_number"].to_numpy(dtype=np.int64)
        features = group[feature_columns].to_numpy(dtype=np.float32)

        segment_start = 0

        # قسّم الفيديو إلى مقاطع متصلة حسب أرقام الإطارات الأصلية
        for i in range(1, len(frames) + 1):
            at_end = i == len(frames)
            has_gap = (
                not at_end
                and frames[i] != frames[i - 1] + 1
            )

            if at_end or has_gap:
                segment_features = features[segment_start:i]
                segment_frames = frames[segment_start:i]

                # نوافذ غير متداخلة: 0-29، 30-59، ...
                for start in range(
                    0,
                    len(segment_features) - WINDOW_SIZE + 1,
                    WINDOW_SIZE,
                ):
                    end = start + WINDOW_SIZE

                    X.append(segment_features[start:end])
                    y.append(label)
                    sources.append(source)
                    video_ids.append(video_id)
                    start_frames.append(int(segment_frames[start]))

                segment_start = i

    if not X:
        return (
            np.empty(
                (0, WINDOW_SIZE, len(feature_columns)),
                dtype=np.float32,
            ),
            np.array([], dtype=str),
            np.array([], dtype=str),
            np.array([], dtype=str),
            np.array([], dtype=np.int64),
        )

    return (
        np.stack(X).astype(np.float32),
        np.array(y, dtype=str),
        np.array(sources, dtype=str),
        np.array(video_ids, dtype=str),
        np.array(start_frames, dtype=np.int64),
    )


def make_training_weights(sources, labels):
    """
    يعطي كل مجموعة مصدر + فئة وزنًا إجماليًا متساويًا في التدريب.
    الأوزان مخصصة للاستخدام مع model.fit(..., sample_weight=...).
    """
    group_counts = Counter(zip(sources, labels))

    weights = np.array(
        [
            1.0 / group_counts[(source, label)]
            for source, label in zip(sources, labels)
        ],
        dtype=np.float32,
    )

    # اجعل متوسط الأوزان 1 لتسهيل استخدامها أثناء التدريب
    weights /= weights.mean()
    return weights


def transform_sequences(scaler, X):
    if len(X) == 0:
        return X.astype(np.float32)

    original_shape = X.shape
    flat = X.reshape(-1, original_shape[-1])
    scaled = scaler.transform(flat)

    return scaled.reshape(original_shape).astype(np.float32)


def print_distribution(title, sources, labels):
    print(f"\n{title}")

    if len(labels) == 0:
        print("  لا توجد تسلسلات.")
        return

    counts = Counter(zip(sources, labels))
    for (source, label), count in sorted(counts.items()):
        print(f"  {source:40} | {label:22} | {count}")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"لم أجد ملف الميزات: {INPUT_CSV.resolve()}"
        )

    data = pd.read_csv(INPUT_CSV)

    required_columns = {
        "video_id",
        "label",
        "frame_number",
        "timestamp_ms",
    }
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            "أعمدة مطلوبة غير موجودة في CSV: "
            + ", ".join(sorted(missing))
        )

    feature_columns = [
        column
        for column in data.columns
        if column not in METADATA_COLUMNS
    ]

    if len(feature_columns) != 78:
        raise ValueError(
            f"توقعت 78 ميزة، لكن وجدت {len(feature_columns)}. "
            "تأكدي من أعمدة CSV."
        )

    data["video_id"] = data["video_id"].astype(str)
    data["label"] = data["label"].astype(str)
    data["frame_number"] = pd.to_numeric(
        data["frame_number"],
        errors="raise",
    ).astype(np.int64)

    data[feature_columns] = data[feature_columns].apply(
        pd.to_numeric,
        errors="raise",
    )

    if not np.isfinite(data[feature_columns].to_numpy()).all():
        raise ValueError("توجد قيم NaN أو لا نهائية في ميزات CSV.")

    # جدول الفيديوهات: كل فيديو يجب أن يكون له تصنيف واحد
    labels_per_video = data.groupby("video_id")["label"].nunique()
    invalid_videos = labels_per_video[labels_per_video != 1].index.tolist()

    if invalid_videos:
        raise ValueError(
            "بعض الفيديوهات لها أكثر من تصنيف. أمثلة: "
            + ", ".join(invalid_videos[:5])
        )

    video_table = (
        data.groupby("video_id", sort=False)
        .agg(label=("label", "first"))
        .reset_index()
    )
    video_table["source"] = video_table["video_id"].apply(get_source)

    # إذا كانت بيانات الاختبار الخارجي موجودة في CSV، افصلها عن التقسيم الداخلي
    external_mask = (
        video_table["source"].str.lower()
        == EXTERNAL_TEST_SOURCE.lower()
    )
    external_video_ids = video_table.loc[
        external_mask, "video_id"
    ].tolist()

    main_videos = video_table.loc[~external_mask].copy()

    if main_videos.empty:
        raise ValueError("لم أجد فيديوهات للتدريب.")

    # تقسيم الفيديوهات 70/15/15 مع المحافظة على توزيع الفئات
    train_val_videos, test_videos = train_test_split(
        main_videos,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=main_videos["label"],
    )

    train_videos, val_videos = train_test_split(
        train_val_videos,
        test_size=0.1765,
        random_state=RANDOM_STATE,
        stratify=train_val_videos["label"],
    )

    print("عدد الفيديوهات:")
    print(f"  Train: {len(train_videos)}")
    print(f"  Validation: {len(val_videos)}")
    print(f"  Internal test: {len(test_videos)}")
    print(f"  External test: {len(external_video_ids)}")

    # تكوين التسلسلات بعد تقسيم الفيديوهات
    X_train, y_train_text, train_sources, train_video_ids, train_starts = (
        create_sequences(
            data,
            train_videos["video_id"].tolist(),
            feature_columns,
        )
    )

    X_val, y_val_text, val_sources, val_video_ids, val_starts = (
        create_sequences(
            data,
            val_videos["video_id"].tolist(),
            feature_columns,
        )
    )

    X_test, y_test_text, test_sources, test_video_ids, test_starts = (
        create_sequences(
            data,
            test_videos["video_id"].tolist(),
            feature_columns,
        )
    )

    X_external, y_external_text, external_sources, external_video_ids_seq, external_starts = (
        create_sequences(
            data,
            external_video_ids,
            feature_columns,
        )
    )

    if len(X_train) == 0:
        raise ValueError("لم ينتج أي تسلسل تدريب كامل من 30 إطارًا.")

    # ترميز التصنيفات؛ يتعلم المحول من بيانات التدريب فقط
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_text)

    known_classes = set(encoder.classes_)

    for name, labels in [
        ("Validation", y_val_text),
        ("Test", y_test_text),
        ("External test", y_external_text),
    ]:
        unknown = set(labels) - known_classes
        if unknown:
            raise ValueError(
                f"{name} يحتوي تصنيفات غير موجودة في التدريب: {unknown}"
            )

    y_val = encoder.transform(y_val_text)
    y_test = encoder.transform(y_test_text)

    if len(y_external_text):
        y_external = encoder.transform(y_external_text)
    else:
        y_external = np.array([], dtype=np.int64)

    # الـScaler يتعلم من إطارات التدريب فقط
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    X_train = transform_sequences(scaler, X_train)
    X_val = transform_sequences(scaler, X_val)
    X_test = transform_sequences(scaler, X_test)
    X_external = transform_sequences(scaler, X_external)

    # موازنة أوزان مصادر التدريب حسب المصدر والفئة
    sample_weight_train = make_training_weights(
        train_sources,
        y_train_text,
    )

    print_distribution(
        "توزيع تسلسلات التدريب",
        train_sources,
        y_train_text,
    )
    print_distribution(
        "توزيع تسلسلات التحقق",
        val_sources,
        y_val_text,
    )
    print_distribution(
        "توزيع الاختبار الداخلي",
        test_sources,
        y_test_text,
    )

    print("\nأشكال البيانات:")
    print("  X_train:", X_train.shape)
    print("  X_val:", X_val.shape)
    print("  X_test:", X_test.shape)
    print("  X_external_test:", X_external.shape)
    print("  الفئات:", list(encoder.classes_))

    np.savez_compressed(
        OUTPUT_NPZ,
        X_train=X_train,
        y_train=y_train,
        sample_weight_train=sample_weight_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        X_external_test=X_external,
        y_external_test=y_external,
        classes=encoder.classes_,
        feature_columns=np.array(feature_columns),
        train_sources=train_sources,
        train_video_ids=train_video_ids,
        train_start_frames=train_starts,
        val_sources=val_sources,
        val_video_ids=val_video_ids,
        val_start_frames=val_starts,
        test_sources=test_sources,
        test_video_ids=test_video_ids,
        test_start_frames=test_starts,
        external_test_sources=external_sources,
        external_test_video_ids=external_video_ids_seq,
        external_test_start_frames=external_starts,
    )

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print(f"\nتم حفظ البيانات في: {OUTPUT_NPZ.resolve()}")
    print(f"تم حفظ الـScaler في: {SCALER_PATH.resolve()}")
    print(f"تم حفظ محول التصنيفات في: {ENCODER_PATH.resolve()}")


if __name__ == "__main__":
    main()