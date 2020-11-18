import cv2
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
from libfaceid.liveness import FaceLivenessModels, FaceLiveness

WINDOW_NAME = "Facial_Recognition"

INPUT_DIR_MODEL_DETECTION  = "models/detection/"
INPUT_DIR_MODEL_ENCODING   = "models/encoding/"
INPUT_DIR_MODEL_TRAINING   = "models/training/"
INPUT_DIR_MODEL_ESTIMATION = "models/estimation/"
INPUT_DIR_MODEL_LIVENESS   = "models/liveness/"


webcam_index = 0
camera = cv2.VideoCapture(webcam_index)
face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)
face_liveness = FaceLiveness(model=FaceLivenessModels.DEFAULT , path=INPUT_DIR_MODEL_ESTIMATION)
face_liveness2 = FaceLiveness(model=FaceLivenessModels.COLORSPACE_YCRCBLUV, path=INPUT_DIR_MODEL_LIVENESS)


def label_face(frame, face_rect, face_id, confidence):
    (x, y, w, h) = face_rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
    if face_id is not None:
        if confidence is not None:
            text = "{} {:.2f}%".format(face_id, confidence)
        else:
            text = "{}".format(face_id)
        cv2.putText(frame, text, (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)




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



def monitor_mouth_opening(mouth_open, mouth_ratio, total_mouth_opens, mouth_counter, mouth_continuous_open):
    if mouth_open:
        #print("mouth more than threshold {:.2f}".format(mouth_ratio))
        mouth_counter += 1
    else:
        #print("mouth:{:.2f} opens:{}".format(mouth_ratio, total_mouth_opens))
        if mouth_counter >= mouth_continuous_open:
            total_mouth_opens += 1
        mouth_counter = 0
    return total_mouth_opens, mouth_counter



face_id, confidence = (None, 0)

eyes_close, eyes_ratio = (False, 0)
total_eye_blinks, eye_counter, eye_continuous_close = (0, 0, 1) # eye_continuous_close should depend on frame rate
mouth_open, mouth_ratio = (False, 0)
total_mouth_opens, mouth_counter, mouth_continuous_open = (0, 0, 1) # eye_continuous_close should depend on frame rate
blinking = 0


while True:
    retval, frame = camera.read()
    faces = face_detector.detect(frame)
    for (index, face) in enumerate(faces):

        #Check if eyes are close and if mouth is open
        eyes_close, eyes_ratio = face_liveness.is_eyes_close(frame, face)
        mouth_open, mouth_ratio = face_liveness.is_mouth_open(frame, face)

        #Detect if frame is a print attack or replay attack based on colorspace
        is_fake_print  = face_liveness2.is_fake(frame, face)
        is_fake_replay = face_liveness2.is_fake(frame, face, flag=1)

        # Identify face only if it is not fake and eyes are open and mouth is close
        # if is_fake_print or is_fake_replay:
        #     face_id, confidence = ("Fake", None)
        # elif not eyes_close and not mouth_open:
        #     face_id, confidence = face_encoder.identify(frame, face)
        #
        # print(face_id)
        # label_face(frame, face, face_id, confidence)

    # Monitor eye blinking and mouth opening for liveness detection
    total_eye_blinks, eye_counter = monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, 1)
    total_mouth_opens, mouth_counter = monitor_mouth_opening(mouth_open, mouth_ratio, total_mouth_opens, mouth_counter, 1)
    if total_eye_blinks > blinking:
        blinking = total_eye_blinks
        print("Blinking")

    cv2.imshow(WINDOW_NAME, frame)
    # Check for user actions
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

print("total_eye_blinks        = {}".format(total_eye_blinks))        # fake face if 0

camera.release()
cv2.destroyAllWindows()