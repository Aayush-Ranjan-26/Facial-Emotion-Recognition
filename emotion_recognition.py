"""Real-time facial emotion recognition — classic ML, no deep learning.

Pipeline:  webcam frame -> Haar face detect -> grayscale -> resize 48x48
           -> HOG features -> SVM classifier -> emotion label

Run:  python emotion_recognition.py
      python emotion_recognition.py --test   (self-check, no dataset/webcam needed)

First run: trains on data/<split>/<emotion>/*.png (FER-2013 folder layout),
prints real accuracy/precision/recall/F1/confusion matrix, saves
emotion_model.pkl, then opens the webcam. Later runs just load the saved
model and go straight to the webcam.
"""
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / "data" / "train"
TEST_DIR = BASE_DIR / "data" / "test"
MODEL_PATH = BASE_DIR / "emotion_model.pkl"

IMG_SIZE = 48  # matches the FER-2013 paper's 48x48 grayscale preprocessing

# Emotion classes = whatever subfolders the dataset actually has (sorted for a
# stable, repeatable label order). Avoids hardcoding a class list that could
# drift from the real data on disk.
LABELS = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir()) if TRAIN_DIR.exists() else []

# SVM training time on libsvm's RBF kernel grows roughly with the SQUARE of the
# sample count. The full training set (~28.7k images) would take well over an
# hour to fit. Capping per-class samples keeps a full run to a few minutes
# while still training and testing on real, unmodified FER-2013 images.
# ponytail: fixed cap for speed, not accuracy tuning. Raise it if you have
# time to spare and want a few extra accuracy points.
MAX_PER_CLASS = 1200


def load_dataset(split_dir, max_per_class=None):
    """Read every image under split_dir/<emotion>/ into (images, labels).

    Supports the FER-2013 "one folder per emotion" layout, which is what this
    project's dataset/ already uses.
    """
    images, labels = [], []
    for label in LABELS:
        folder = split_dir / label
        if not folder.exists():
            continue
        files = sorted(folder.iterdir())
        if max_per_class:
            files = files[:max_per_class]
        for path in files:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # grayscale: emotion
            if img is None:                                    # is in shape, not color
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    return images, labels


def extract_features(images):
    """Turn each grayscale image into a HOG feature vector.

    HOG (Histogram of Oriented Gradients) summarizes the direction of edges in
    small cells across the image -- e.g. the angle of a raised eyebrow or a
    curved mouth -- which is what actually distinguishes expressions. It
    throws away things that don't matter here, like exact lighting or skin
    tone, giving the SVM a compact, comparable vector per face instead of
    2304 raw, noisy pixel values.
    """
    features = []
    for img in images:
        normalized = img.astype("float32") / 255.0  # 0-255 -> 0-1, standard scale for HOG
        feat = hog(normalized, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2))
        features.append(feat)
    return np.array(features)


def train_model():
    print("Loading dataset...")
    train_images, train_labels = load_dataset(TRAIN_DIR, MAX_PER_CLASS)
    test_images, test_labels = load_dataset(TEST_DIR)
    print(f"  train: {len(train_images)} images, test: {len(test_images)} images")
    print(f"  classes: {LABELS}")

    print("Extracting HOG features...")
    X_train = extract_features(train_images)
    X_test = extract_features(test_images)

    # SVM finds the boundary that best separates the emotion classes in HOG
    # feature space. An RBF kernel lets that boundary curve, since emotions
    # aren't linearly separable in raw HOG space. class_weight="balanced"
    # compensates for "disgust" having far fewer training images than the rest.
    print("Training SVM (a few minutes)...")
    model = SVC(kernel="rbf", probability=True, class_weight="balanced")
    model.fit(X_train, train_labels)

    evaluate_model(model, X_test, test_labels)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def evaluate_model(model, X_test, y_test):
    """Print real accuracy/precision/recall/F1/confusion matrix on the test set."""
    predictions = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nPer-class precision / recall / F1:")
    print(classification_report(y_test, predictions, labels=LABELS))
    print(f"Confusion matrix (rows=true, cols=predicted), label order {LABELS}:")
    print(confusion_matrix(y_test, predictions, labels=LABELS))


def preprocess_face(gray_face):
    """Same steps as training: resize to 48x48, normalize, extract HOG.

    Training and webcam inference must use identical preprocessing, or the
    SVM is being asked to classify features it never learned the shape of.
    """
    face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    return hog(face, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2))


def run_realtime(model):
    # Ships inside opencv-python -- no separate file to manage or download.
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        sys.exit("Could not open webcam.")

    print("Webcam started. Press Q to quit.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                     minSize=(48, 48))
            # detectMultiScale returns an empty list when no face is visible --
            # the loop below simply does nothing that frame, no crash, no spam.
            for (x, y, w, h) in faces:
                features = preprocess_face(gray[y:y + h, x:x + w]).reshape(1, -1)
                emotion = model.predict(features)[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion.upper(), (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Facial Emotion Recognition (press Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


def self_check():
    """Smallest check that fails if preprocessing breaks -- no dataset or webcam needed."""
    fake_face = np.random.randint(0, 256, (30, 20), dtype=np.uint8)  # odd, non-square crop
    feat_a = preprocess_face(fake_face)
    feat_b = preprocess_face(fake_face)
    assert feat_a.ndim == 1, "HOG features must be a flat vector for the SVM"
    assert np.array_equal(feat_a, feat_b), "same input must give the same features"
    print(f"ok: preprocess_face -> {feat_a.shape[0]}-d feature vector")


def main():
    if MODEL_PATH.exists():
        print(f"Loading saved model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    else:
        if not TRAIN_DIR.exists():
            sys.exit(f"Dataset not found at {TRAIN_DIR} -- expected data/train/<emotion>/*.png")
        model = train_model()
    run_realtime(model)


if __name__ == "__main__":
    self_check() if "--test" in sys.argv else main()
