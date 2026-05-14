import cv2
import os
import pandas as pd
import time
import numpy as np

def capture_student_faces_generator(name, roll, department):
    """Capture images for a new student and yield frames for web streaming."""
    if not name or not roll or not department:
        yield b'Error: All fields required'
        return

    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        yield b'Error: Haar cascade not found'
        return

    face_detector = cv2.CascadeClassifier(cascade_path)

    csv_file = "Student_data/student_details.csv"
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            student_id = 1
        else:
            student_id = int(df['ID'].max()) + 1
    except FileNotFoundError:
        df = pd.DataFrame(columns=["ID", "Name", "Roll", "Department"])
        student_id = 1

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        yield b'Error: Could not access webcam'
        return

    count = 0

    while count < 60:
        ret, img = cam.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Optimize detection speed by using scaleFactor 1.3
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            
            # Save the captured face
            face_img_path = f"Student_data/{name}.{student_id}.{count}.jpg"
            cv2.imwrite(face_img_path, gray[y:y + h, x:x + w])
            
            # Add text showing progress
            cv2.putText(img, f"Capturing: {count}/60", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Brief pause to not overwhelm the system and allow UI to keep up
            time.sleep(0.05)

        # Encode frame as JPEG to yield
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()

    if count >= 60:
        new_record = pd.DataFrame([{"ID": student_id, "Name": name, "Roll": roll, "Department": department}])
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(csv_file, index=False)
        
    # Send a final frame indicating completion to prevent UI from freezing
    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_img, "Capture Complete!", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    ret, buffer = cv2.imencode('.jpg', blank_img)
    frame = buffer.tobytes()
    # Yield a few times to ensure the browser gets it
    for _ in range(5):
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)
