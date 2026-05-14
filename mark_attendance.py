import cv2
import pandas as pd
import os
import datetime
import time

# Global variable to control the generator loop
stop_attendance_flag = False

def mark_attendance_generator():
    global stop_attendance_flag
    stop_attendance_flag = False

    """Start webcam, recognize faces, and yield frames."""
    if not os.path.exists("Trained_model/trainer.yml"):
        yield b'Error: Model not trained'
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("Trained_model/trainer.yml")
    except Exception as e:
        yield b'Error: Could not load model'
        return

    cascade_path = "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    # Load student details
    csv_file = "Student_data/student_details.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        yield b'Error: Student details not found'
        return

    # Prepare daily attendance file
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    attendance_file = f"Attendance/Attendance_{date_str}.csv"
    
    if not os.path.exists(attendance_file):
        att_df = pd.DataFrame(columns=["Name", "Roll", "Department", "Date", "Time", "Status"])
        att_df.to_csv(attendance_file, index=False)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        yield b'Error: Could not access webcam'
        return

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Track who has been marked
    marked_students = set()
    
    # Load already marked students from today
    try:
        today_att = pd.read_csv(attendance_file)
        marked_students.update(today_att["Roll"].tolist())
    except Exception:
        pass

    while not stop_attendance_flag:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use scaleFactor 1.3 for faster detection
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Predict the face
            student_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # 85 is a much better threshold for LBPH than 50
            if confidence < 85:
                student_row = df[df['ID'] == student_id]
                if not student_row.empty:
                    name = student_row.iloc[0]['Name']
                    roll = student_row.iloc[0]['Roll']
                    dept = student_row.iloc[0]['Department']
                    
                    display_text = f"{name} ({roll})"
                    
                    # Mark attendance
                    if roll not in marked_students:
                        time_str = datetime.datetime.now().strftime("%H:%M:%S")
                        new_record = pd.DataFrame([{
                            "Name": name, 
                            "Roll": roll, 
                            "Department": dept, 
                            "Date": date_str, 
                            "Time": time_str, 
                            "Status": "Present"
                        }])
                        att_df = pd.read_csv(attendance_file)
                        att_df = pd.concat([att_df, new_record], ignore_index=True)
                        att_df.to_csv(attendance_file, index=False)
                        marked_students.add(roll)
                else:
                    name = "Unknown"
                    display_text = name
            else:
                name = "Unknown"
                display_text = "Unknown"
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.putText(img, display_text, (x, y - 10), font, 0.8, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()

def stop_attendance():
    global stop_attendance_flag
    stop_attendance_flag = True
