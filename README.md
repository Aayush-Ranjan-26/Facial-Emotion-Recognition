# Facial Emotion Recognition

Terminal Python app. Webcam -> Haar cascade face detection -> HOG features ->
SVM -> emotion label drawn on the video feed. No web app, no server, no
deep learning -- classic ML, one file.

Pipeline:

```
webcam frame -> face detect (Haar) -> grayscale -> resize 48x48
             -> HOG features -> SVM classifier -> emotion label
```

Trained and evaluated on FER-2013 (`data/train/<emotion>/*.png`,
`data/test/<emotion>/*.png`), 7 classes: angry, disgust, fear, happy,
neutral, sad, surprise.

## Setup

```
.venv\Scripts\pip install -r requirements.txt
```

## Run

```
.venv\Scripts\python emotion_recognition.py --test   # self-check, no dataset/webcam needed
.venv\Scripts\python emotion_recognition.py           # train (first run) or load, then webcam, Q to quit
```

First run trains the SVM on `data/`, prints real accuracy/precision/recall/F1/
confusion matrix, and saves `emotion_model.pkl`. Later runs load that file
directly and skip straight to the webcam.

## Notes

- `MAX_PER_CLASS` in `emotion_recognition.py` caps training images per class
  (1200). SVM training time grows roughly with the square of the sample
  count, so the full ~28.7k-image training set would take over an hour; the
  cap keeps a full run to a few minutes. Raise it if you have time to spare.
- The Haar cascade ships inside `opencv-python` (`cv2.data.haarcascades`) --
  no separate `.xml` file to keep in the repo.
