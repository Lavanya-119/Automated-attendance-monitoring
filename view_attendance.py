import pandas as pd
import os
import datetime

def get_todays_attendance():
    """Return a list of dictionaries containing today's attendance records."""
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    attendance_file = f"Attendance/Attendance_{date_str}.csv"
    
    if not os.path.exists(attendance_file):
        return []
        
    try:
        df = pd.read_csv(attendance_file)
        # Convert to list of dictionaries
        return df.to_dict('records')
    except Exception as e:
        print(f"Error reading attendance file: {e}")
        return []
