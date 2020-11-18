import os
import enum
from libfaceid.detector import FaceDetectorModels, FaceDetector
from libfaceid.encoder import FaceEncoderModels, FaceEncoder
from libfaceid.classifier import FaceClassifierModels



# spacifying path clause
INPUT_DIR_DATASET = "datasets"
INPUT_DIR_MODEL_DETECTION = "models/detection/"
INPUT_DIR_MODEL_ENCODING = "models/encoding/"
INPUT_DIR_MODEL_TRAINING = "models/training/"



# creating enumerations using class
class Detector(enum.Enum):
    haarcascade = FaceDetectorModels.HAARCASCADE
    dlibhog = FaceDetectorModels.DLIBHOG
    dlibcnn = FaceDetectorModels.DLIBCNN
    ssdresnet = FaceDetectorModels.SSDRESNET
    mtcnn = FaceDetectorModels.MTCNN
    facenet = FaceDetectorModels.FACENET

class Encoder(enum.Enum):
    lbph = FaceEncoderModels.LBPH
    openface = FaceEncoderModels.OPENFACE
    dlibresnet = FaceEncoderModels.DLIBRESNET
    facenet = FaceEncoderModels.FACENET

class Classifier(enum.Enum):
    nbayes = FaceClassifierModels.NAIVE_BAYES
    lsvm = FaceClassifierModels.LINEAR_SVM
    rvfsvm = FaceClassifierModels.RBF_SVM
    nneigh = FaceClassifierModels.NEAREST_NEIGHBORS
    dtree = FaceClassifierModels.DECISION_TREE
    rforest = FaceClassifierModels.RANDOM_FOREST
    nn = FaceClassifierModels.NEURAL_NET
    aboost = FaceClassifierModels.ADABOOST
    qda =  FaceClassifierModels.QDA



# checking the path
def ensure_directory(file_path):
    directory = os.path.dirname("./" + file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)



# taking names fromn dataset
def get_dataset_names(file_path):
    for (_d, names, _f) in os.walk(file_path):
        return names
    return None



# training the model
def train_recognition(model_detector, model_encoder, model_classifier, verify):
    ensure_directory(INPUT_DIR_DATASET)

    print("")
    names = get_dataset_names(INPUT_DIR_DATASET)
    if names is not None:
        print("Names " + str(names))
        for name in names:
            for (_d, _n, files) in os.walk(INPUT_DIR_DATASET + "/" + name):
                print(name + ": " + str(files))
    print("")

    ensure_directory(INPUT_DIR_MODEL_TRAINING)
    face_detector = FaceDetector(model=model_detector, path=INPUT_DIR_MODEL_DETECTION)
    face_encoder = FaceEncoder(model=model_encoder, path=INPUT_DIR_MODEL_ENCODING,
                               path_training=INPUT_DIR_MODEL_TRAINING, training=True)
    face_encoder.train(face_detector, path_dataset=INPUT_DIR_DATASET, verify=verify, classifier=model_classifier)
    print("train_recognition completed")



# training wrapper function
def training(detector, encoder, classifier):
    train_recognition(detector, encoder, classifier, True)
    print("\nImage dataset training completed!")



####################### checking the xahid_training.py ####################
training(Detector.mtcnn, Encoder.facenet, Classifier.nn)
########################################################################