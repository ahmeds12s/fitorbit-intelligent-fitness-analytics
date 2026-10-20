import csv #writing the data in csv file , which every frame represent record and the columns represent features
import math #operating mathematical operation 
from collections import Counter, defaultdict #n o elements , n of videos and frames that is accepted or declined 
#Automatically create statistical counters for each data source, without needing to configure each key manually.
from pathlib import Path #dealing with paths of files and folder in order

import cv2 #OpenCV for dealing with videos and read frames , changing its color
import mediapipe as mp #To detect the position of the body and its joints.


PROJECT_DIR = Path( #the main path for the project
    r"C:\Users\AlHuda\Desktop\AI WorkSpace\work-base Projects\S1 L3\FitOrbit"
)

DATA_DIR = PROJECT_DIR / "archive_4" #data path
MODEL_PATH = PROJECT_DIR / "models" / "pose_landmarker_lite.task" #Location of the pre-trained MediaPipe file for pose detection.
OUTPUT_CSV = PROJECT_DIR / "paper_frame_features_v2.csv" #The storage location for the resulting Features file.

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"} #A collection of file extensions that the program will accept. The presence of an extension within this collection allows the code to determine whether a given video file is supported.
MIN_VISIBILITY = 0.5 #A joint is considered valid if its visibility value is greater than or equal to 0.5.


# MediaPipe pose landmark indices , Each key contains the name of a specific landmark, and the value is the corresponding landmark number in MediaPipe.
LANDMARKS = { # 22 landmark.
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
}

#Each tuple here represents 3 joints, for which we will calculate the angle.
ANGLE_TRIPLETS = [
    ("left_hip", "left_shoulder", "left_elbow"),
    ("right_hip", "right_shoulder", "right_elbow"),
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
    ("left_shoulder", "left_hip", "left_knee"),
    ("right_shoulder", "right_hip", "right_knee"),
    ("left_knee", "left_ankle", "left_heel"),
    ("right_knee", "right_ankle", "right_heel"),
    ("left_ankle", "left_heel", "left_foot_index"),
    ("right_ankle", "right_heel", "right_foot_index"),
]


#This part defines the essential joints that must be visible for the frame to be accepted.
REQUIRED_JOINTS = {
    "barbell biceps curl": ("shoulder", "elbow", "wrist"),
    "push-up": ("shoulder", "elbow", "hip"),
    "shoulder press": ("shoulder", "elbow", "wrist"),
    "squat": ("hip", "knee"),
} #The code requires that the hip and knee be visible on at least one side—either the right or the left.


def get_label(video_path: Path):
   #The exercise classification is derived from the name of the parent folder.
    label = video_path.parent.name.strip().lower()
    return label if label in REQUIRED_JOINTS else None 

#It allows the code to collect statistics for each source individually, even if the videos are located across many subfolders.
def get_source(video_path: Path) -> str:
    """اسم مجموعة البيانات، مثل final_kaggle أو my_test_video_1."""
    relative = video_path.relative_to(DATA_DIR)
    return relative.parts[0] if relative.parts else "unknown"

#This function will be used in later stages to determine whether or not the joint is acceptable.
def is_visible(point) -> bool:
    return getattr(point, "visibility", 0.0) >= MIN_VISIBILITY

#It determines whether a specific part of the body contains the essential joints required for the exercise.
def side_is_valid(points, label: str, side: str) -> bool:
    """نقبل الإطار إذا ظهرت المفاصل الأساسية على اليمين أو اليسار."""
    for joint in REQUIRED_JOINTS[label]:
        landmark_name = f"{side}_{joint}"
        landmark_index = LANDMARKS[landmark_name]

        if not is_visible(points[landmark_index]):
            return False #If any required joint is not clear, the function immediately returns False, without needing to continue checking the remaining joints.

    return True #It will only be executed if all the required joints on this side pass the inspection.


def get_xyz(points, landmark_name: str):
    #It returns the coordinates and places zeros for the undefined points.
    point = points[LANDMARKS[landmark_name]]

    if not is_visible(point):
        return 0.0, 0.0, 0.0

    return float(point.x), float(point.y), float(point.z) #If the point is clear, the three coordinates are returned as decimal numbers.


def calculate_angle(point_a, point_b, point_c) -> float:
    """حساب الزاوية ABC بالدرجات."""
    vector_ba = [point_a[i] - point_b[i] for i in range(3)] #This line calculates the vector from point \(B\) to point \(A\).
    vector_bc = [point_c[i] - point_b[i] for i in range(3)] #It calculates the vector from \(B\) to \(C\).
    #Calculating the length of each vector
    norm_ba = math.sqrt(sum(value * value for value in vector_ba))
    norm_bc = math.sqrt(sum(value * value for value in vector_bc))

    if norm_ba == 0.0 or norm_bc == 0.0:
        return 0.0 #If one of the two points coincides with point \(B\), the length of one of the vectors becomes zero, and we cannot divide by it.
        #Therefore, the function returns zero instead of encountering a mathematical error.

    cosine = sum( #Calculating the cosine of an angle
        vector_ba[i] * vector_bc[i] for i in range(3)
    ) / (norm_ba * norm_bc)

    cosine = max(-1.0, min(1.0, cosine)) #This line confines it to the correct scope. between 1 , -1
    return math.degrees(math.acos(cosine)) #math.acos: Calculates the angle in radians. math.degrees: Converts it into degrees.

#Creating CSV file columns
def make_header():
    columns = [
        "video_id",
        "label",
        "frame_number",
        "timestamp_ms",
    ]
    #Creating coordinate columns
    for landmark_name in LANDMARKS:
        columns.extend([
            f"{landmark_name}_x",
            f"{landmark_name}_y",
            f"{landmark_name}_z", 
        ])
    #Creating corner columns
    for angle in ANGLE_TRIPLETS:
        columns.append("angle_" + "_".join(angle))

    # 4 definition/timing columns + 66 coordinates + 12 angles = 82
    assert len(columns) == 82
    return columns


def make_frame_row(video_id, label, frame_number, timestamp_ms, points):
   # The primary joints must be visible on at least one side.
    if not (
        side_is_valid(points, label, "left")
        or side_is_valid(points, label, "right")
    ):
        return None #This means that no valid data row will be written for this frame.
    #Extract the coordinates of each joint
    coordinates = {
        landmark_name: get_xyz(points, landmark_name)
        for landmark_name in LANDMARKS
    }

    row = [
        video_id,
        label,
        frame_number,
        timestamp_ms,
    ]
    #Adding joint coordinates
    for landmark_name in LANDMARKS:
        row.extend(coordinates[landmark_name])

    for angle_points in ANGLE_TRIPLETS:
        if all(#The all() function returns True if all three joints are visible.
            is_visible(points[LANDMARKS[name]])
            for name in angle_points
        ):
            angle = calculate_angle(
                coordinates[angle_points[0]],
                coordinates[angle_points[1]],
                coordinates[angle_points[2]],
            )
        else:
            angle = 0.0 #It records zero instead of calculating the angle.

        row.append(angle)

    assert len(row) == 82 #He confirms that the row contains 82 values
    return row

#The function responsible for executing most of the pipeline steps on a single video
def process_video(video_path: Path, writer, stats):
    label = get_label(video_path)
    source = get_source(video_path)

    if label is None: #Ignore unsupported exercises
        stats[source]["unsupported"] += 1
        return

    video_id = str(video_path.relative_to(DATA_DIR).with_suffix(""))
    capture = cv2.VideoCapture(str(video_path))

    #If OpenCV fails to open the video, the code logs the failure, prints a message, and then stops processing that video.
    if not capture.isOpened():
        stats[source]["cannot_open"] += 1
        print(f"[تعذر الفتح] {video_path}")
        return

    stats[source]["opened"] += 1 #It increments the counter for videos that have been opened.

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1, #The code requires the detection of a maximum of one person per frame.
    )

    frame_number = 0
    saved_frames = 0
    pose_frames = 0
    rejected_frames = 0

    try: #The `try` block allows the code to handle errors during the video processing stage.
        # A new Landmarker for each video to restart tracking and timing.
        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:
            #The first line reads the frame rate of the video; for example, a video with a frame rate of 30 FPS means there are 30 frames per second.
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            #This is a loop that repeatedly reads frames until the video ends or the reading process fails.
            while True:
                success, frame = capture.read()
                if not success:
                    break

                timestamp_ms = int(frame_number * 1000 / fps) #Frame timing calculation

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #OpenCV typically reads images in BGR color order, whereas the code prepares them in the RGB order required here for MediaPipe.
                mp_image = mp.Image( #Here, the image is converted into an `mp.Image` object, with its format specified as SRGB.
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                result = landmarker.detect_for_video( #This is the line that actually invokes the MediaPipe model on the current frame.
                    mp_image,
                    timestamp_ms,
                )
                #There is no "pose"—the code does not create a new row, nor does it increment `rejected_frames`. This type of rejection is specifically for frames where a body was detected, but the joint-related criteria were not met.
                if result.pose_landmarks:
                    pose_frames += 1
                    points = result.pose_landmarks[0]

                    row = make_frame_row(
                        video_id,
                        label,
                        frame_number,
                        timestamp_ms,
                        points,
                    )

                    if row is None:
                        rejected_frames += 1
                    else:
                        writer.writerow(row)
                        saved_frames += 1

                frame_number += 1

    except Exception as error:
        stats[source]["errors"] += 1
        print(f"[خطأ] {video_path}: {error}")

    finally:
        capture.release()

    stats[source]["processed"] += 1
    stats[source]["frames_read"] += frame_number
    stats[source]["pose_detected_frames"] += pose_frames
    stats[source]["rejected_frames"] += rejected_frames
    stats[source]["saved_frames"] += saved_frames
    stats[source][f"label:{label}"] += 1

    print(
        f"[اكتمل] {video_id} | "
        f"إطارات مقروءة: {frame_number} | "
        f"إطارات محفوظة: {saved_frames} | "
        f"مرفوضة بسبب المفاصل: {rejected_frames}"
    )


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"مجلد الداتا غير موجود: {DATA_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ملف النموذج غير موجود: {MODEL_PATH}")

    video_files = sorted(
    path
    for path in DATA_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in VIDEO_EXTENSIONS
    and get_source(path).lower() != "my_test_video_1"
    )

    if not video_files:
        raise RuntimeError(f"لم أجد فيديوهات داخل: {DATA_DIR}")

    candidate_counts = Counter(
        (get_source(path), get_label(path) or "unsupported")
        for path in video_files
    )

    print(f"إجمالي ملفات الفيديو: {len(video_files)}")
    print("الفيديوهات الموجودة حسب المصدر والتصنيف:")
    for (source, label), count in sorted(candidate_counts.items()):
        print(f"  {source} | {label}: {count}")

    stats = defaultdict(Counter)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(make_header())

        for video_path in video_files:
            process_video(video_path, writer, stats)

    print("\nملخص النتائج:")
    for source in sorted(stats):
        print(f"\n{source}:")
        for key, value in sorted(stats[source].items()):
            print(f"  {key}: {value}")

    print(f"\nملف النتائج: {OUTPUT_CSV}")
    print(
        "\nملاحظة: فيديوهات my_test_video_1 ستظهر في الملف؛ "
        "اتركيها للاختبار النهائي ولا تدخليها في التدريب."
    )


if __name__ == "__main__":
    main()