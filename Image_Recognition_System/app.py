from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from model_utils import predict_image_class

app = Flask(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file uploaded", 400
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)  # Sanitize filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {filepath}")
        file.save(filepath)  # Save file to disk

        predictions = predict_image_class(filepath)
        return render_template('index.html', image=filename, predictions=predictions)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)