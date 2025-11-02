import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime
import base64
from werkzeug.utils import secure_filename
from model import predict_emotion
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create static folder if it doesn't exist
os.makedirs('static', exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  timestamp TEXT NOT NULL,
                  image_path TEXT NOT NULL,
                  predicted_emotion TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get('name', 'Anonymous')
        image_source = request.form.get('source')  # 'upload' or 'webcam'
        
        # Handle image based on source
        if image_source == 'upload':
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            file = request.files['file']
            if file.filename == '' or file.filename is None:
                return jsonify({'error': 'No file selected'}), 400
            
            # Generate safe filename
            original_filename = secure_filename(file.filename) if file.filename else 'image.jpg'
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
        elif image_source == 'webcam':
            # Get base64 image from webcam
            image_data = request.form.get('image')
            if not image_data:
                return jsonify({'error': 'No webcam image provided'}), 400
            
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            
            # Convert to image
            image = Image.open(io.BytesIO(image_bytes))
            filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
        else:
            return jsonify({'error': 'Invalid image source'}), 400
        
        # Predict emotion
        emotion = predict_emotion(filepath)
        
        # Save to database
        conn = sqlite3.connect('emotions.db')
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO predictions (name, timestamp, image_path, predicted_emotion) VALUES (?, ?, ?, ?)",
                  (name, timestamp, filepath, emotion))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'name': name,
            'timestamp': timestamp
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)