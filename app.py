from flask import Flask, render_template, request, Response, flash, redirect, url_for
from utils import setup
from capture_faces import capture_student_faces_generator
from train_model import train_lbph_model
from mark_attendance import mark_attendance_generator, stop_attendance
from view_attendance import get_todays_attendance
import os
import pandas as pd

# Run initial setup to ensure directories exist
setup()

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_attendance_app'

# Global state to pass data to the generator
register_data = {"name": None, "roll": None, "department": None, "active": False}
mark_data = {"active": False}

@app.route('/')
def index():
    csv_file = "Student_data/student_details.csv"
    total_students = 0
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            total_students = len(df)
        except:
            pass
            
    todays_attendance = get_todays_attendance()
    present_today = len(todays_attendance)
    
    return render_template('index.html', total_students=total_students, present_today=present_today)

@app.route('/register', methods=['GET', 'POST'])
def register_student():
    if request.method == 'POST':
        name = request.form.get('name')
        roll = request.form.get('roll')
        department = request.form.get('department')
        
        register_data["name"] = name
        register_data["roll"] = roll
        register_data["department"] = department
        register_data["active"] = True
        
        return render_template('register.html', active_capture=True)
        
    return render_template('register.html', active_capture=False)

@app.route('/video_feed_register')
def video_feed_register():
    if not register_data["active"]:
        return "Not active", 400
        
    def generate():
        for frame in capture_student_faces_generator(
            register_data["name"], 
            register_data["roll"], 
            register_data["department"]):
            yield frame
        # Once done, set active to False
        register_data["active"] = False

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        success, message = train_lbph_model()
        if success:
            flash(message, 'success')
        else:
            flash(message, 'danger')
        return redirect(url_for('train'))
        
    return render_template('train.html')

@app.route('/mark')
def mark_att():
    mark_data["active"] = True
    return render_template('mark.html')

@app.route('/stop_mark')
def stop_mark():
    stop_attendance()
    mark_data["active"] = False
    flash("Stopped attendance marking.", "info")
    return redirect(url_for('index'))

@app.route('/video_feed_attendance')
def video_feed_attendance():
    if not mark_data["active"]:
        return "Not active", 400
        
    return Response(mark_attendance_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view')
def view_att():
    attendance_records = get_todays_attendance()
    return render_template('view.html', attendance_records=attendance_records)

if __name__ == '__main__':
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
