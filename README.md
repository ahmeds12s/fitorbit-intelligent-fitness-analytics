> Academic Name : Intelligent Fitness Analytics and Personalized Workout Generation System 
                               > commercial Name : FitOrbit


> DataSet Link : https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset
> GitHub Repo : https://github.com/RiccardoRiccio/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting#project-structure
  * An extension of the previous 'Fitness-AI-Coach': a complete web application with real-time exercise recognition and counting.

> Research Paper : https://arxiv.org/pdf/2411.11548
> Progress Data : https://drive.google.com/drive/folders/1wzJ0V0ACEWbwZdSeQbRnxRS1I_lwLyaV?usp=drive_link

 > Project Steps
     Project Pipeline
1. Organize the videos
   Place each video in a folder named after its exercise, such as squat or push-up. Keep my_test_video_1 separate for final testing.
2. Extract body landmarks
   Use MediaPipe to detect body landmarks in each video. Save the landmark coordinates and joint angles to a CSV file.
3. Check the extracted data
   Make sure all expected exercise classes and video sources appear in the output. Investigate videos or frames that were skipped.
4. Create 30-frame sequences
   Sort frames by video and frame number, then group them into sequences of 30 frames. Each frame has 78 model features.
5. Split the training data
   Split videos into training, validation, and internal test sets. Keep all frames from each video in the same set.
6. Scale the features
   Fit a StandardScaler on the training data, then use it to scale the validation and test data. Do not use video_id, label, frame_number, or timestamp_ms as model features.
7. Train the model
   Train an LSTM or BiLSTM model to classify each sequence as an exercise.
8. Evaluate the model
   Use the validation set to tune the model. After training is complete, evaluate it on the separate my_test_video_1 videos.
9. Test on new videos
   Run the model on videos it has not seen before to check how well it works in practice.

> our progress until now : 

> extract features
The project has completed pose landmark and feature extraction from the training datasets. #10/18/2026
- Extracted body landmarks and joint angles using MediaPipe.
- Saved 173,957 valid frames to paper_frame_features_v2.csv.
- Each row represents one video frame and contains 78 movement features, plus video ID, exercise label, frame number, and timestamp.
- Processed four exercise classes from final_kaggle_with_additional_video, similar_dataset, and synthetic_dataset.
- Kept my_test_video_1 separate for final evaluation.
Next steps: split the data by video, create 30-frame sequences, balance source weights, and train and evaluate the BiLSTM model.


> Sequence Preparation and Preprocessing

The extracted frame-level features are prepared for training the BiLSTM model:

- Excluded metadata columns (`video_id`, `label`, `frame_number`, and `timestamp_ms`) from the model input.
- Split videos into training, validation, and internal test sets **before** creating sequences, preventing frames from the same video from appearing in multiple sets.
- Created non-overlapping sequences of 30 consecutive frames. Sequences do not cross gaps in frame numbers.
- Used the training set to fit the `LabelEncoder` and `StandardScaler`, then applied the saved transformations to the other sets.
- Created training sample weights to balance the contribution of each data source and exercise class.
- Kept `my_test_video_1` separate as external test data when it is present in the feature CSV.

The preprocessing script saves the processed arrays to `preprocessed_data.npz` and saves the fitted preprocessing objects to `scaler.pkl` and `label_encoder.pkl`.

In the latest run, the data contained 3,677 training sequences, 834 validation sequences, and 755 internal test sequences. Each sequence has 30 frames and 78 features.


> BiLSTM Model Training and Evaluation

A Bidirectional LSTM model was trained to classify four exercises from 30-frame pose sequences, each containing 78 features.

- The model uses two Bidirectional LSTM layers with 73 units, followed by dropout and a softmax classification layer.
- Training uses source-and-class sample weights to reduce the effect of dataset imbalance.
- Early stopping, best-model checkpointing, and learning-rate reduction callbacks are used during training.
- The best model is selected using validation loss, then evaluated on the internal test set.

On the latest internal test run, the model achieved **97.35% accuracy** with a loss of **0.0613**.

Training saves the model as `fit_orbit_bilstm.keras`. It also exports the training history, classification report, and confusion matrix to the `training_results` folder.

The external test set (`my_test_video_1`) is not evaluated by this training script when it is empty in the preprocessed data. It should be prepared and evaluated separately.

> External Video Testing — In Progress

An external testing pipeline has been prepared for videos in `my_test_video_1`. It extracts pose features, groups consecutive frames into non-overlapping 30-frame sequences, applies the scaler and label encoder saved during training, and evaluates the saved BiLSTM model.

The pipeline is designed to export per-sequence predictions, a classification report, and a confusion matrix to `training_results`.

Before external evaluation can be completed, the test feature definitions must be aligned with the training pipeline. The current test script produces 81 features, while the trained model expects 78, and the angle definitions and ordering must also match training.


> External Test Evaluation

The saved BiLSTM model is evaluated on the preprocessed external test sequences stored in `preprocessed_test.npz`. The script loads the existing model and label encoder without retraining, then calculates loss, accuracy, precision, recall, and F1-score.

It saves the results to the `training_results` folder:

- `external_test_classification_report.txt`
- `external_test_confusion_matrix.csv`
- `external_test_predictions.csv`

The predictions file includes each sequence’s true label, predicted label, whether the prediction was correct, and—when available—the source video and starting frame.

> Live Camera API

FitOrbit provides a FastAPI WebSocket endpoint for live exercise classification. The browser streams camera frames to the API, which uses MediaPipe Pose Landmarker to extract 22 body landmarks and 12 joint angles.

The API collects 30 valid frames, applies the saved training scaler, and sends the BiLSTM model’s predicted exercise and confidence scores back to the browser. It loads the trained model, scaler, and label encoder without retraining.

To run the API:

```bash
python -m pip install "uvicorn[standard]"
python -m uvicorn live_api:app --host 127.0.0.1 --port 8000
