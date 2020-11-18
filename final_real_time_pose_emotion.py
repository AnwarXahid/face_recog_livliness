import sys
import argparse
import cv2
import datetime
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
from libfaceid.pose import FacePoseEstimatorModels, FacePoseEstimator
from libfaceid.age import FaceAgeEstimatorModels, FaceAgeEstimator
from libfaceid.gender import FaceGenderEstimatorModels, FaceGenderEstimator
from libfaceid.emotion import FaceEmotionEstimatorModels, FaceEmotionEstimator



# Set the window name
WINDOW_NAME = "Facial_Recognition"

# Set the input directories
INPUT_DIR_DATASET               = "datasets"
INPUT_DIR_MODEL_DETECTION       = "models/detection/"
INPUT_DIR_MODEL_ENCODING        = "models/encoding/"
INPUT_DIR_MODEL_TRAINING        = "models/training/"
INPUT_DIR_MODEL_ESTIMATION      = "models/estimation/"




def cam_init(cam_index):
    cap = cv2.VideoCapture(cam_index)
    return cap


def label_face(frame, face_rect, face_id, confidence):
    (x, y, w, h) = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
    if face_id is not None:
        cv2.putText(frame, "{} {:.2f}%".format(face_id, confidence),
            (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def process_facedetection(model_detector, model_poseestimator, model_ageestimator, model_genderestimator, model_emotionestimator, cam_index):

    # Initialize the camera
    camera = cam_init(cam_index)

    try:
        # Initialize face detection
        face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION, minfacesize=120)
        # Initialize face pose/age/gender estimation
        face_pose_estimator = FacePoseEstimator(model=model_poseestimator, path=INPUT_DIR_MODEL_ESTIMATION)
        face_age_estimator = FaceAgeEstimator(model=model_ageestimator, path=INPUT_DIR_MODEL_ESTIMATION)
        face_gender_estimator = FaceGenderEstimator(model=model_genderestimator, path=INPUT_DIR_MODEL_ESTIMATION)
        face_emotion_estimator = FaceEmotionEstimator(model=model_emotionestimator, path=INPUT_DIR_MODEL_ESTIMATION)
    except:
        print("Warning, check if models and trained dataset models exists!")
    (age, gender, emotion) = (None, None, None)


    while (True):

        # Capture frame from webcam
        ret, frame = camera.read()
        if frame is None:
            print("Error, check if camera is connected!")
            break


        # Detect and identify faces in the frame
        faces = face_detector.detect(frame)
        for (index, face) in enumerate(faces):
            (x, y, w, h) = face

            # Detect age, gender, emotion
            face_image = frame[y:y+h, h:h+w]
            age = face_age_estimator.estimate(frame, face_image)
            gender = face_gender_estimator.estimate(frame, face_image)
            emotion = face_emotion_estimator.estimate(frame, face_image)

            # Detect and draw face pose locations
            shape = face_pose_estimator.detect(frame, face)
            face_pose_estimator.add_overlay(frame, shape)

            # Display age, gender, emotion
            if True: # Added condition to easily disable text
                cv2.putText(frame, "Age: {}".format(age),
                    (x, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Gender: {}".format(gender),
                    (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Emotion: {}".format(emotion),
                    (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        # Display the resulting frame
        cv2.imshow(WINDOW_NAME, frame)

        # Check for user actions
        keyPressed = cv2.waitKey(1) & 0xFF
        if keyPressed == 27: # ESC
            break
        elif keyPressed == 13: # Enter
            cv2.imwrite(WINDOW_NAME + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg", frame);


    # Release the camera
    camera.release()
    cv2.destroyAllWindows()


def run(cam_index):

#    detector=FaceDetectorModels.HAARCASCADE
#    detector=FaceDetectorModels.DLIBHOG
#    detector=FaceDetectorModels.DLIBCNN
#    detector=FaceDetectorModels.SSDRESNET
    detector=FaceDetectorModels.MTCNN
#    detector=FaceDetectorModels.FACENET

    encoder=FaceEncoderModels.LBPH
#    encoder=FaceEncoderModels.OPENFACE
#    encoder=FaceEncoderModels.DLIBRESNET
#    encoder=FaceEncoderModels.FACENET

    poseestimator    = FacePoseEstimatorModels.DLIB68
    ageestimator     = FaceAgeEstimatorModels.CV2CAFFE
    genderestimator  = FaceGenderEstimatorModels.CV2CAFFE
    emotionestimator = FaceEmotionEstimatorModels.KERAS

    process_facedetection(
        detector,
        poseestimator,
        ageestimator,
        genderestimator,
        emotionestimator,
        cam_index)



######### running main function #########
run(0)

