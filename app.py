import re
import time
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
from Category import Category
from FitnessClass import FitnessClass
from model import process, generate_feedback

app = Flask(__name__)
app.secret_key = "its_the_football_guys"
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

football_classes = [
    FitnessClass(1, "Throwing a Football", "Adithya Rajesh", "45 mins", "football-icon.png", "uploads/test2.mov"),
    FitnessClass(2, "Snapping a Football", "Adithya Rajesh", "1 hour", "football-icon.png", "Football_snap.mov"),
    FitnessClass(2, "Running Routes", "Eshan Jaffar", "1.5 hours", "football-icon.png", "Football_snap.mov"),
    FitnessClass(2, "Evading Defenders", "Pranav Eega", "2 hours", "football-icon.png", "Football_snap.mov"),
    FitnessClass(2, "Tackling Safely", "Preetam Jukalkar", "20 minutes", "football-icon.png", "Football_snap.mov"),
]

baseball_classes = [
    FitnessClass(3, "Squat", "Pranav Eega", "15 mins", "gym.png", "Squat.mov"),
    FitnessClass(4, "Bicep Curl", "Adithya Rajesh", "30 mins", "gym.png", "Bicep_curl.mov"),
    FitnessClass(5, "Jumping Jacks", "Adithya Rajesh", "30 mins", "gym.png", "Jumping_Jacks.mov"),
    FitnessClass(5, "Bench Press", "Eshan Jaffar", "45 mins", "gym.png", "Jumping_Jacks.mov"),
    FitnessClass(5, "Cardio", "Preetam Jukalkar", "1 hr", "gym.png", "Jumping_Jacks.mov"),
    FitnessClass(5, "Push Ups", "Pranav Eega", "15 mins", "gym.png", "Jumping_Jacks.mov")
]

basketball_classes = [
    FitnessClass(6, "Chicken Dance", "Adithya Rajesh", "30 mins", "dance.png", "chicken_dance.mov"),
    FitnessClass(9, "Sticky", "Tyler, the Creator", "1 hr", "dance.png", "Sticky.MOV"),
    FitnessClass(7, "Sprinkler", "Adithya Rajesh", "15 mins", "dance.png", "Sprinkler.MOV"),
    FitnessClass(8, "Wave", "Adithya Rajesh", "10 mins", "dance.png", "Wave.mov")
]

categories = [
    Category("Upload", "upload.png", "https://www.youtube.com/embed/kNoy1tOY_Kw?autoplay=1", football_classes,
                [], '', ''),
    Category("Football", "football.png", "https://www.youtube.com/embed/kNoy1tOY_Kw?autoplay=1", football_classes,
                [], '', ''),
    Category("Gym", "gym.png", "https://www.youtube.com/embed/b1H3xO3x_Js?autoplay=1", baseball_classes,
                [], '', ''),
    Category("Dance", "dance.png", "Sticky.MOV", basketball_classes,
                [], '', ''),
    Category("Baseball", "baseball.png", "https://www.youtube.com/embed/KDorKy-13ak?autoplay=1", basketball_classes,
                [], '', ''),
    Category("Basketball", "basketball.png", "https://www.youtube.com/embed/KDorKy-13ak?autoplay=1", basketball_classes,
                [], '', ''),
    Category("Tennis", "tennis.png", "https://www.youtube.com/embed/KDorKy-13ak?autoplay=1", basketball_classes,
                [], '', ''),
    Category("Soccer", "soccer.png", "https://www.youtube.com/embed/KDorKy-13ak?autoplay=1", basketball_classes,
                [], '', ''),
    Category("Cricket", "cricket.png", "https://www.youtube.com/embed/KDorKy-13ak?autoplay=1", basketball_classes,
                [], '', ''),
    Category("Hockey", "hockey.png", "https://www.youtube.com/embed/KDorKy-13ak?autoplay=1", basketball_classes,
                [], '', ''),
    Category("Badminton", "badminton.png", "https://www.youtube.com/embed/KDorKy-13ak?autoplay=1", basketball_classes,
                [], '', ''),
    Category("Golf", "golf.png", "https://www.youtube.com/embed/KDorKy-13ak?autoplay=1", basketball_classes,
                [], '', ''),
    Category("Thraxxing", "dance.png", "https://www.youtube.com/embed/KDorKy-13ak?autoplay=1", basketball_classes,
                [], '', '')
]

@app.route('/')
def index():
    return render_template('index.html', categories=categories)

@app.route('/classes')
def home():
    active_category_name = session.get('active_category', categories[0].name)
    active_category = next((c for c in categories if c.name == active_category_name), categories[0])
    return render_template('classes.html', categories=categories, active_category=active_category)

@app.route('/set_active_category', methods=['POST'])
def set_active_category():
    data = request.get_json()
    session['active_category'] = data.get('category')
    return jsonify({"success": True})

@app.route('/class/<int:class_id>')
def class_detail(class_id):
    for category in categories:
        for fitness_class in category.classes:
            if fitness_class.id == class_id:
                return render_template('class_detail.html', fitness_class=fitness_class)
    return "Class not found", 404

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']
        print(file1, file2)

        if file1 and file2:
            file1.save(os.path.join(UPLOAD_FOLDER, file1.filename))
            file2.save(os.path.join(UPLOAD_FOLDER, file2.filename))
            session['file1'] = os.path.join(UPLOAD_FOLDER, file1.filename)
            session['file2'] = os.path.join(UPLOAD_FOLDER, file2.filename)
            return redirect(url_for("process_page"))
        else:
            return 'Missing files!', 400
    return render_template('upload.html')

def process_videos():
    file1_path = session['file1']
    file2_path = session['file2']
    print(file1_path, file2_path)
    process(file1_path, file2_path)

    frames_folder = os.path.join("static", "uploads", "frames")

    # Detect bad frames
    bad_frames_data = detect_bad_frame_sections(frames_folder)
    low_score_segments = bad_frames_data.get("bad_sections", [])

    # Your Gemini API Key (replace with your actual key or load from environment variables)
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # Generate feedback using the provided function
    feedback_text = generate_feedback(
        user_video_name=file1_path,
        ref_video_name=file2_path,
        low_score_segments=low_score_segments,
        gemini_api_key=gemini_api_key
    )

    print("Videos processed successfully!")
    return feedback_text

@app.route('/process')
def process_page():
    frames_folder = os.path.join("static", "uploads", "frames")  # Adjust path as needed
    bad_frames_data = detect_bad_frame_sections(frames_folder)  # Detect bad frame sections

    output_path = os.path.join("static", "uploads", "output.mp4")

    feedback_text = process_videos()

    return render_template('process.html', output_path=output_path, bad_sections=bad_frames_data["bad_sections"], response_text = feedback_text)

def detect_bad_frame_sections(frames_folder):
    """
    Scan a directory for bad frames and detect sections with 8+ consecutive bad frames.

    :param frames_folder: Path to the folder containing frames
    :return: JSON object with bad frame sections
    """
    frame_pattern = re.compile(r"(\d+)_output_bad\.jpeg")
    frames = []

    # Scan directory and extract bad frames
    for filename in sorted(os.listdir(frames_folder), key=lambda x: int(re.search(r"(\d+)", x).group())):
        match = frame_pattern.match(filename)
        if match:
            frame_num = int(match.group(1))
            frames.append(frame_num)

    # Identify consecutive bad frame sections
    bad_sections = []
    current_section = []

    for i in range(len(frames)):
        if i == 0 or frames[i] == frames[i - 1] + 1:
            current_section.append(frames[i])
        else:
            if len(current_section) >= 20:  # Change the condition to >= 8
                bad_sections.append((current_section[0], current_section[-1]))
            current_section = [frames[i]]

    # Add last section if it qualifies
    if len(current_section) >= 20:  # Change the condition to >= 8
        bad_sections.append((current_section[0], current_section[-1]))

    return {"bad_sections": bad_sections}

if __name__ == '__main__':
    app.run(debug=True)