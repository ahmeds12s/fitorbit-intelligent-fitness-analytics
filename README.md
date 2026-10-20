> Academic Name : Intelligent Fitness Analytics and Personalized Workout Generation System 
                               > commercial Name : FitOrbit


> DataSet Link : https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset
> GitHub Repo : https://github.com/RiccardoRiccio/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting#project-structure
  * An extension of the previous 'Fitness-AI-Coach': a complete web application with real-time exercise recognition and counting.

> Research Paper : https://arxiv.org/pdf/2411.11548

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

Current Progress
The project has completed pose landmark and feature extraction from the training datasets. #10/18/2026
- Extracted body landmarks and joint angles using MediaPipe.
- Saved 173,957 valid frames to paper_frame_features_v2.csv.
- Each row represents one video frame and contains 78 movement features, plus video ID, exercise label, frame number, and timestamp.
- Processed four exercise classes from final_kaggle_with_additional_video, similar_dataset, and synthetic_dataset.
- Kept my_test_video_1 separate for final evaluation.
Next steps: split the data by video, create 30-frame sequences, balance source weights, and train and evaluate the BiLSTM model.
