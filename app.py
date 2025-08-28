import json
from flask import Flask, render_template, request, redirect, url_for
import random
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the palmistry knowledge base
with open('palmistry_knowledge_base.json', 'r') as f:
    knowledge_base = json.load(f)

def analyze_palm(image_path):
    """
    Simulates palm analysis by randomly selecting characteristics from the knowledge base.
    """
    analysis = {}
    for line, meanings in knowledge_base.items():
        # Select one or two random characteristics for each line
        num_characteristics = random.randint(1, 2)
        selected_keys = random.sample(list(meanings.keys()), num_characteristics)
        analysis[line] = {key: meanings[key] for key in selected_keys}
    return analysis

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        analysis_results = analyze_palm(filepath)

        return render_template('results.html', results=analysis_results)

if __name__ == '__main__':
    app.run(debug=True)
