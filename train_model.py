import cv2
import numpy as np
import os
from PIL import Image

def train_lbph_model():
    """Train the LBPH face recognizer using captured images."""
    path = 'Student_data'
    
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        return False, "OpenCV contrib is not installed. Please install opencv-contrib-python."

    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        return False, f"Haar cascade file not found: {cascade_path}"

    detector = cv2.CascadeClassifier(cascade_path)

    def get_images_and_labels(path):
        if not os.path.exists(path):
            return [], []
            
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        face_samples = []
        ids = []

        for image_path in image_paths:
            PIL_img = Image.open(image_path).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            try:
                id_str = os.path.split(image_path)[-1].split(".")[1]
                student_id = int(id_str)
            except (IndexError, ValueError):
                continue

            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(student_id)

        return face_samples, ids

    faces, ids = get_images_and_labels(path)

    if not faces or not ids:
        return False, "No training data found. Register a student first."

    recognizer.train(faces, np.array(ids))

    if not os.path.exists('Trained_model'):
        os.makedirs('Trained_model')

    recognizer.write('Trained_model/trainer.yml')
    return True, f"Model trained successfully on {len(np.unique(ids))} student(s)."
