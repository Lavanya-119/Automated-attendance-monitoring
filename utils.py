import os
import pandas as pd

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        "Student_data",
        "Trained_model",
        "Attendance",
        "output_images",
        "UI_Image"
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def initialize_student_csv():
    """Create the student details CSV if it doesn't exist."""
    csv_file = "Student_data/student_details.csv"
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=["ID", "Name", "Roll", "Department"])
        df.to_csv(csv_file, index=False)

def download_haarcascade():
    """Download Haar Cascade file if it doesn't exist."""
    file_name = "haarcascade_frontalface_default.xml"
    if not os.path.exists(file_name):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        import urllib.request
        try:
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(url, file_name)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download Haar cascade file: {e}")

def setup():
    """Run all setup tasks."""
    ensure_directories()
    initialize_student_csv()
    download_haarcascade()
