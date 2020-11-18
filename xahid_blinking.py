import cv2
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
from libfaceid.liveness import FaceLivenessModels, FaceLiveness


INPUT_DIR_MODEL_DETECTION  = "models/detection/"
INPUT_DIR_MODEL_ENCODING   = "models/encoding/"
INPUT_DIR_MODEL_TRAINING   = "models/training/"
INPUT_DIR_MODEL_ESTIMATION = "models/estimation/"
INPUT_DIR_MODEL_LIVENESS   = "models/liveness/"


camera = cv2.VideoCapture(0)
face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)
face_liveness = FaceLiveness(model=FaceLivenessModels.DEFAULT , path=INPUT_DIR_MODEL_ESTIMATION)
face_liveness2 = FaceLiveness(model=FaceLivenessModels.COLORSPACE_YCRCBLUV, path=INPUT_DIR_MODEL_LIVENESS)


def monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, eye_continuous_close):
    if eyes_close:
        #print("eye less than threshold {:.2f}".format(eyes_ratio))
        eye_counter += 1
    else:
        #print("eye:{:.2f} blinks:{}".format(eyes_ratio, total_eye_blinks))
        if eye_counter >= eye_continuous_close:
            total_eye_blinks += 1
        eye_counter = 0
    return total_eye_blinks, eye_counter




eyes_close, eyes_ratio = (False, 0)
total_eye_blinks, eye_counter, eye_continuous_close = (0, 0, 0.5) # eye_continuous_close should depend on frame rate
blinking = 0


while True:
    retval, frame = camera.read()
    faces = face_detector.detect(frame)
    for (index, face) in enumerate(faces):
        eyes_close, eyes_ratio = face_liveness.is_eyes_close(frame, face)
        mouth_open, mouth_ratio = face_liveness.is_mouth_open(frame, face)

    # Monitor eye blinking and mouth opening for liveness detection
    total_eye_blinks, eye_counter = monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, 1)
    if total_eye_blinks > blinking:
        blinking = total_eye_blinks
        print("Blinking : {}".format(blinking))

    cv2.imshow("Facial_Recognition", frame)
    # Check for user actions
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

print("total_eye_blinks        = {}".format(total_eye_blinks))

camera.release()
cv2.destroyAllWindows()