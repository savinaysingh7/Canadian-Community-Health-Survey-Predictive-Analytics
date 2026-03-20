import sqlite3
import json
from datetime import datetime

DB_NAME = "health_ai.db"

def init_db():
    """Initialize the database with required tables."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  role TEXT DEFAULT 'doctor')''')
    
    # Patients Table
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  age_group INTEGER,
                  gender INTEGER,
                  created_at TEXT)''')
    
    # Assessments Table
    c.execute('''CREATE TABLE IF NOT EXISTS assessments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id INTEGER,
                  date TEXT,
                  input_data TEXT,
                  risk_score REAL,
                  prediction INTEGER,
                  FOREIGN KEY(patient_id) REFERENCES patients(id))''')
    
    # Wellbeing Assessments Table
    c.execute('''CREATE TABLE IF NOT EXISTS wellbeing_assessments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id INTEGER,
                  date TEXT,
                  input_data TEXT,
                  wellbeing_score REAL,
                  FOREIGN KEY(patient_id) REFERENCES patients(id))''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# --- User Operations ---
def create_user(username, password_hash, role='doctor'):
    try:
        conn = get_db_connection()
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                     (username, password_hash, role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def get_user(username):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return user

# --- Patient Operations ---
def add_patient(name, age_group, gender):
    conn = get_db_connection()
    cur = conn.execute("INSERT INTO patients (name, age_group, gender, created_at) VALUES (?, ?, ?, ?)",
                       (name, age_group, gender, datetime.now().isoformat()))
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid

def get_all_patients():
    conn = get_db_connection()
    patients = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    conn.close()
    return patients

def get_patient(pid):
    conn = get_db_connection()
    patient = conn.execute("SELECT * FROM patients WHERE id = ?", (pid,)).fetchone()
    conn.close()
    return patient

# --- Assessment Operations ---
def log_assessment(patient_id, input_data, risk_score, prediction):
    conn = get_db_connection()
    conn.execute("INSERT INTO assessments (patient_id, date, input_data, risk_score, prediction) VALUES (?, ?, ?, ?, ?)",
                 (patient_id, datetime.now().isoformat(), json.dumps(input_data), risk_score, int(prediction)))
    conn.commit()
    conn.close()

def log_wellbeing_assessment(patient_id, input_data, wellbeing_score):
    conn = get_db_connection()
    conn.execute("INSERT INTO wellbeing_assessments (patient_id, date, input_data, wellbeing_score) VALUES (?, ?, ?, ?)",
                 (patient_id, datetime.now().isoformat(), json.dumps(input_data), float(wellbeing_score)))
    conn.commit()
    conn.close()

def get_patient_history(patient_id):
    conn = get_db_connection()
    history = conn.execute("SELECT * FROM assessments WHERE patient_id = ? ORDER BY date DESC", (patient_id,)).fetchall()
    conn.close()
    return history

def get_wellbeing_history(patient_id):
    conn = get_db_connection()
    history = conn.execute("SELECT * FROM wellbeing_assessments WHERE patient_id = ? ORDER BY date DESC", (patient_id,)).fetchall()
    conn.close()
    return history

def get_all_assessments():
    conn = get_db_connection()
    data = conn.execute("SELECT * FROM assessments").fetchall()
    conn.close()
    return data

def get_all_wellbeing_assessments():
    conn = get_db_connection()
    data = conn.execute("SELECT * FROM wellbeing_assessments").fetchall()
    conn.close()
    return data
