from flask import Flask, render_template, request, jsonify, session
from Category import Category
from FitnessClass import FitnessClass

app = Flask(__name__)
app.secret_key = "its_the_football_guys"

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

@app.route('/upload')
def upload():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)