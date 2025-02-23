import os

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from Category import Category
from FitnessClass import FitnessClass
from model import process

app = Flask(__name__)
app.secret_key = "its_the_football_guys"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if it doesn't exist

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}


football_classes = [
    FitnessClass("Throwing a Football", "John Doe", "1 hour", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw"),
    FitnessClass("Touching a football", "Doe Doe", "2 hour", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw"),
    FitnessClass("Snapping a football", "John John", "3 hour", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw")
]

baseball_classes = [
    FitnessClass("Throwing a baseball", "John Doe", "1 hour", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw"),
    FitnessClass("Touching a baseball", "Doe Doe", "2 hour", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw"),
    FitnessClass("Snapping a baseball", "John John", "3 hour", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw")
]

basketball_classes = [
    FitnessClass("Throwing a basketball_classes", "John Doe", "1 hour", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw"),
    FitnessClass("Touching a basketball_classes", "Doe Doe", "2 hour", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw"),
    FitnessClass("Snapping a basketball_classes", "John John", "3 hour", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw")
]

categories = [
    Category("Football", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw", football_classes),
    Category("Baseball", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw", baseball_classes),
    Category("Basketball", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw", basketball_classes)
]

@app.route('/')
def home():
    active_category_name = session.get('active_category', categories[0])  # Default to first category
    active_category = next((c for c in categories if c.name == active_category_name), categories[0])
    return render_template('index.html', categories=categories, active_category=active_category)

@app.route('/set_active_category', methods=['POST'])
def set_active_category():
    data = request.get_json()
    session['active_category'] = data.get('category')
    return jsonify({"success": True})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload')
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    file1_url = None
    file2_url = None

    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            return "Missing files", 400

        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1.filename == '' or file2.filename == '':
            return "No selected files", 400

        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

            file1.save(file1_path)
            file2.save(file2_path)

            # Store file paths in session for later processing
            session["file1"] = file1_path
            session["file2"] = file2_path

            file1_url = f"/uploads/{filename1}"
            file2_url = f"/uploads/{filename2}"

            # After successful upload, redirect to a processing page
            return redirect(url_for("process_page"))

    return render_template('upload.html', file1_url=file1_url, file2_url=file2_url)


# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

import shutil

@app.route('/process_page', methods=['GET', 'POST'])
def process_page():
    file1_path = session.get("file1")
    file2_path = session.get("file2")

    if not file1_path or not file2_path:
        return "Upload both videos first!", 400

    if request.method == "GET":
        filename1 = os.path.basename(file1_path)
        filename2 = os.path.basename(file2_path)
        file1_url = url_for("uploaded_file", filename=filename1)
        file2_url = url_for("uploaded_file", filename=filename2)
        return render_template("process.html", file1_url=file1_url, file2_url=file2_url,
                               file1_path=file1_path, file2_path=file2_path)

    # Run processing (this will generate 'output.mp4' somewhere)
    process(file1_path, file2_path)

    # Define the expected output path
    output_video_path = "output.mp4"  # Change this if stored elsewhere
    if not os.path.exists(output_video_path):
        return "Error: Processed video not found!", 500

    # Move it to uploads folder
    final_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.mp4")
    shutil.move(output_video_path, final_path)

    # Store in session
    session["processed_video"] = os.path.abspath(final_path)

    return redirect(url_for("final_output"))

# Final output page to show the processed video.
@app.route('/final_output')
@app.route('/final_output')
def final_output():
    processed_video = session.get("processed_video")

    if not processed_video or not os.path.exists(processed_video):
        return "Processed video not found.", 404

    # Serve from uploads folder
    filename = os.path.basename(processed_video)
    final_url = url_for("uploaded_file", filename=filename)

    return render_template("final_output.html", processed_video_url=final_url)


if __name__ == '__main__':
    app.run(debug=True)