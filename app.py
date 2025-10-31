from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid


# Import model integration function (stub for now)
try:
    from model_integration import predict_emotions
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))


    filename = secure_filename(file.filename)
    filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Mock prediction or real model call
    if MODEL_AVAILABLE:
        dominant, emotions = predict_emotions(filepath)
    else:
        emotions = {
        'happy': 45.0,
        'sad': 20.0,
        'neutral': 25.0,
        'angry': 10.0
        }
        dominant = max(emotions, key=emotions.get)


    return render_template('index.html', image_url=url_for('uploaded_file', filename=filename), 
                           dominant=dominant, emotions=emotions)


if __name__ == '__main__':
    app.run(debug=True)