from flask import Flask, render_template, request, jsonify, session
from Category import Category
from FitnessClass import FitnessClass

app = Flask(__name__)
app.secret_key = "its_the_football_guys"

football_classes = [
    FitnessClass(1, "Throwing a Football", "John Doe", "1 hour", "football-icon.png", "Football.mov"),
    FitnessClass(2, "Snapping a football", "Doe Doe", "2 hour", "football-icon.png", "Football_snap.mov"),
]

baseball_classes = [
    FitnessClass(3, "Squat", "John Doe", "1 hour", "football-icon.png", "Squat.MOV"),
    FitnessClass(4, "Bicep Curl", "Doe Doe", "2 hour", "football-icon.png", "Bicep_curl.mov"),
    FitnessClass(5, "Jumping Jacks", "John John", "3 hour", "football-icon.png", "Jumping_Jacks.mov")
]

basketball_classes = [
    FitnessClass(6, "Chicken Dance", "John Doe", "1 hour", "football-icon.png", "chicken_dance.mov"),
    FitnessClass(7, "Sprinkler", "Doe Doe", "2 hour", "football-icon.png", "Sprinkler.MOV"),
    FitnessClass(8, "Wave", "John John", "3 hour", "football-icon.png", "Wave.mov")
]

categories = [
    Category("Football", "football-icon.png", "https://www.youtube.com/embed/kNoy1tOY_Kw", football_classes),
    Category("Gym", "pngtree-gym-facility-for-the-customers-at-five-star-hotels-png-image_4704890.png", "https://www.youtube.com/embed/B4kNiCWTl7", baseball_classes),
    Category("Dance", "10580.png", "https://www.youtube.com/embed/ntS0hequ388?si=qxta1i2n1CbRFIPO", basketball_classes)
]

@app.route('/')
def home():
    active_category_name = session.get('active_category', categories[0].name)
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