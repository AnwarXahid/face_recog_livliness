# Face Recognition with Liveness Detection

An automated employee attendance system using face recognition with anti-spoofing capabilities for my previous company. This system eliminates the need for fingerprint scanners or manual attendance marking by providing secure, real-time face authentication with liveness detection.

## Overview

This project implements a comprehensive face recognition system that includes:
- Real-time face detection and recognition
- Liveness detection to prevent spoofing attacks
- Pose and emotion detection capabilities
- Automated attendance tracking
- Web-based interface using Flask

## Features

- **Face Recognition**: Accurate employee identification using trained facial models
- **Anti-Spoofing**: Liveness detection to prevent photo/video spoofing attempts
- **Real-time Processing**: Live webcam-based attendance marking
- **Pose Detection**: Employee pose analysis for better authentication
- **Emotion Recognition**: Optional emotion detection capabilities
- **Web Interface**: Flask-based web application for easy deployment
- **Automated Training**: Tools for training custom face recognition models

## Project Structure

```
face_recog_livliness/
├── datasets/                           # Training datasets and employee photos
├── libfaceid/                         # Core face recognition library
├── models/                            # Trained ML models and weights
├── templates/                         # HTML templates for web interface
├── final_basic_real_time_face_recog.py    # Basic real-time face recognition
├── final_fr_with_liveliness.py           # Face recognition with liveness detection
├── final_real_time_pose_emotion.py       # Pose and emotion detection
├── final_training_model.py               # Model training script
├── training.py                           # Training utilities
├── xahid_training.py                     # Custom training implementation
├── testing_webcam.py                     # Webcam testing utilities
├── testing_webcam_flask.py              # Flask web interface testing
├── testing_webcam_livenessdetection.py  # Liveness detection testing
├── xahid_blinking.py                     # Blink detection for liveness
├── xahid_demo.py                         # Demo implementation
└── requirements.txt                      # Python dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AnwarXahid/face_recog_livliness.git
   cd face_recog_livliness
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up datasets**:
   - Place employee photos in the `datasets/` directory
   - Organize photos by employee ID or name
   - Ensure good quality, well-lit photos for better recognition

## Usage

### Training the Model

1. **Prepare your dataset**:
   ```bash
   python training.py
   ```

2. **Train the face recognition model**:
   ```bash
   python final_training_model.py
   ```

3. **Custom training (alternative)**:
   ```bash
   python xahid_training.py
   ```

### Running the System

1. **Basic Face Recognition**:
   ```bash
   python final_basic_real_time_face_recog.py
   ```

2. **Face Recognition with Liveness Detection**:
   ```bash
   python final_fr_with_liveliness.py
   ```

3. **With Pose and Emotion Detection**:
   ```bash
   python final_real_time_pose_emotion.py
   ```

4. **Web Interface**:
   ```bash
   python testing_webcam_flask.py
   ```

### Testing Components

- **Test webcam functionality**:
  ```bash
  python testing_webcam.py
  ```

- **Test liveness detection**:
  ```bash
  python testing_webcam_livenessdetection.py
  ```

- **Test blink detection**:
  ```bash
  python xahid_blinking.py
  ```

- **Run demo**:
  ```bash
  python xahid_demo.py
  ```

## Key Components

### Liveness Detection
The system implements multiple liveness detection methods:
- **Blink Detection**: Monitors eye blinks to ensure a live person
- **Face Movement**: Tracks subtle facial movements
- **Depth Analysis**: Analyzes facial depth to prevent photo spoofing

### Face Recognition
- Uses deep learning models for accurate face encoding
- Supports multiple employees with high accuracy
- Real-time processing for instant attendance marking

### Anti-Spoofing Features
- Photo spoofing prevention
- Video replay attack detection
- 3D mask detection capabilities

## Configuration

1. **Camera Setup**: Ensure your webcam is properly configured
2. **Model Paths**: Update model paths in configuration files
3. **Employee Database**: Set up employee records with corresponding face encodings
4. **Attendance Logging**: Configure attendance database/file storage

## Performance Optimization

- **GPU Support**: Enable GPU acceleration for faster processing
- **Model Optimization**: Use optimized models for real-time performance
- **Threading**: Implement multi-threading for concurrent processing

## Security Features

- **Anti-Spoofing**: Multiple layers of spoofing detection
- **Secure Storage**: Encrypted storage of face encodings
- **Access Control**: Role-based access to attendance data
- **Audit Trail**: Complete logging of attendance events

## Deployment

For production deployment:

1. **Hardware Requirements**:
   - Good quality webcam (HD recommended)
   - Adequate lighting conditions
   - GPU recommended for real-time processing

2. **Server Deployment**:
   - Deploy Flask application on production s
