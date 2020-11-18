import cv2
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder import FaceEncoderModels, FaceEncoder

# Set the window name
WINDOW_NAME = "Facial_Recognition"

INPUT_DIR_MODEL_DETECTION = "models/detection/"
INPUT_DIR_MODEL_ENCODING = "models/encoding/"
INPUT_DIR_MODEL_TRAINING = "models/training/"


webcam_index = 0
camera = cv2.VideoCapture(webcam_index)
face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING,
                           path_training=INPUT_DIR_MODEL_TRAINING, training=False)



def label_face(frame, face_rect, face_id, confidence):
    (x, y, w, h) = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
    if face_id is not None:
        if confidence is not None:
            text = "{} {:.2f}%".format(face_id, confidence)
        else:
            text = "{}".format(face_id)
        cv2.putText(frame, text, (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)




while True:
    retval, frame = camera.read()
    faces = face_detector.detect(frame)
    for (index, face) in enumerate(faces):
        face_id, confidence = face_encoder.identify(frame, face)
        label_face(frame, face, face_id, confidence)
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()