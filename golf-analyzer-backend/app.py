from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import tempfile
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load("swing_classifier.joblib")
mp_pose = mp.solutions.pose

def extract_pose_features(video_path):
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            landmark_vec = []
            for lm in result.pose_landmarks.landmark:
                landmark_vec.extend([lm.x, lm.y, lm.z])
            features.append(landmark_vec)

    cap.release()
    pose.close()

    return np.mean(features, axis=0) if features else np.zeros(99)

@app.route("/analyze", methods=["POST"])
def analyze():
    video = request.files["video"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video.save(tmp.name)
        features = extract_pose_features(tmp.name)
        os.unlink(tmp.name)

    prediction = model.predict([features])[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)