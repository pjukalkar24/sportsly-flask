from flask import Flask, render_template, request, jsonify, session
from Category import Category
from FitnessClass import FitnessClass

app = Flask(__name__)
app.secret_key = "its_the_football_guys"

football_classes = [
    FitnessClass(1, "Throwing a Football", "John Doe", "1 hour", "football-icon.png", "Football.mov"),
    FitnessClass(2, "Touching a football", "Doe Doe", "2 hour", "football-icon.png", "Squat.mov"),
    FitnessClass(3, "Snapping a football", "John John", "3 hour", "football-icon.png", "Football.mov")
]

baseball_classes = [
    FitnessClass(4, "Throwing a baseball", "John Doe", "1 hour", "football-icon.png", "Football.mov"),
    FitnessClass(5, "Touching a baseball", "Doe Doe", "2 hour", "football-icon.png", "Football.mov"),
    FitnessClass(6, "Snapping a baseball", "John John", "3 hour", "football-icon.png", "Football.mov")
]

basketball_classes = [
    FitnessClass(7, "Throwing a basketball_classes", "John Doe", "1 hour", "football-icon.png", "Football.mov"),
    FitnessClass(8, "Touching a basketball_classes", "Doe Doe", "2 hour", "football-icon.png", "Football.mov"),
    FitnessClass(9, "Snapping a basketball_classes", "John John", "3 hour", "football-icon.png", "Football.mov")
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

@app.route('/class/<int:class_id>')
def class_detail(class_id):
    for category in categories:
        for fitness_class in category.classes:
            if fitness_class.id == class_id:
                return render_template('class_detail.html', fitness_class=fitness_class)
    return "Class not found", 404

@app.route('/upload')
def upload():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)