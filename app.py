import json
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the palmistry knowledge base
with open('palmistry_knowledge_base.json', 'r') as f:
    knowledge_base = json.load(f)

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands

def _find_palm_lines(image, landmarks):
    """
    Finds the three major palm lines in the image using landmarks to define ROIs.
    """
    h, w, _ = image.shape
    lines = {}

    # Convert landmarks to pixel coordinates
    pixel_landmarks = [{'index': lm['index'], 'x': int(lm['x'] * w), 'y': int(lm['y'] * h)} for lm in landmarks]

    # --- Utility function for processing an ROI ---
    def process_roi(roi_coords):
        x1, y1, x2, y2 = roi_coords
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        roi = image[y1:y2, x1:x2]
        if roi.size == 0: return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours: return None

        longest_contour = max(contours, key=cv2.contourArea)

        # Convert points to full image coordinates
        line_points = [{'x': p[0][0] + x1, 'y': p[0][1] + y1} for p in longest_contour]
        return line_points

    # --- Heart Line Detection ---
    p5 = pixel_landmarks[5]  # INDEX_FINGER_MCP
    p17 = pixel_landmarks[17] # PINKY_MCP
    heart_roi_coords = (p5['x'], min(p5['y'], p17['y']) - int(0.05 * h), p17['x'], max(p5['y'], p17['y']) + int(0.15 * h))
    heart_line_points = process_roi(heart_roi_coords)
    if heart_line_points:
        lines['heart_line'] = {'status': 'success', 'points': heart_line_points}
    else:
        lines['heart_line'] = {'status': 'error', 'message': 'Heart line not detected.'}

    # --- Head Line Detection ---
    p1 = pixel_landmarks[1] # THUMB_CMC
    p13 = pixel_landmarks[13] # MIDDLE_FINGER_MCP
    head_roi_y_center = p1['y'] + int((p5['y'] - p1['y']) * 0.5)
    head_roi_coords = (p1['x'], head_roi_y_center - int(0.1 * h), p13['x'], head_roi_y_center + int(0.1 * h))
    head_line_points = process_roi(head_roi_coords)
    if head_line_points:
        lines['head_line'] = {'status': 'success', 'points': head_line_points}
    else:
        lines['head_line'] = {'status': 'error', 'message': 'Head line not detected.'}

    # --- Life Line Detection ---
    # This is more complex due to its curve. We'll use a wider ROI.
    p0 = pixel_landmarks[0] # WRIST
    p2 = pixel_landmarks[2] # THUMB_MCP
    p9 = pixel_landmarks[9] # MIDDLE_FINGER_MCP
    life_roi_coords = (p0['x'] - int(0.1 * w), p9['y'], p2['x'] + int(0.2 * w), p0['y'])
    life_line_points = process_roi(life_roi_coords)
    if life_line_points:
        lines['life_line'] = {'status': 'success', 'points': life_line_points}
    else:
        lines['life_line'] = {'status': 'error', 'message': 'Life line not detected.'}

    return lines


def _extract_line_features(lines, image_dims):
    """
    Analyzes the geometric properties of the detected palm lines.
    """
    features = {}
    w, h = image_dims

    for line_name, line_data in lines.items():
        if line_data['status'] != 'success':
            features[line_name] = {'status': 'error', 'message': f'{line_name} not detected.'}
            continue

        points = np.array([list(p.values()) for p in line_data['points']], dtype=np.int32)

        # 1. Length Analysis
        arc_length = cv2.arcLength(points, closed=False)
        # Normalize length by palm width (a rough heuristic)
        normalized_length = arc_length / w

        length_feature = 'medium'
        if normalized_length > 0.6:
            length_feature = 'long'
        elif normalized_length < 0.3:
            length_feature = 'short'

        # 2. Curvature Analysis
        start_point = points[0]
        end_point = points[-1]
        straight_dist = np.linalg.norm(start_point - end_point)

        curvature_feature = 'straight'
        # Avoid division by zero
        if straight_dist > 0:
            ratio = arc_length / straight_dist
            if ratio > 1.2:
                curvature_feature = 'curved'

        features[line_name] = {
            'status': 'success',
            'length': length_feature,
            'curvature': curvature_feature,
            # 'points': line_data['points'] # Keep points for potential drawing later
        }

    return features


def _map_features_to_reading(features):
    """
    Maps the extracted line features to the human-readable interpretations
    from the knowledge base.
    """
    reading = {}
    for line_name, line_features in features.items():
        if line_features['status'] != 'success':
            # Use a more descriptive name for the UI
            display_name = line_name.replace('_', ' ').title()
            reading[display_name] = {'Error': f'Could not analyze {display_name}.'}
            continue

        line_reading = {}
        # Map length feature
        length_key = f"length_{line_features['length']}"
        if length_key in knowledge_base[line_name]:
            line_reading['Length'] = knowledge_base[line_name][length_key]

        # Map curvature feature
        curvature_key = f"curvature_{line_features['curvature']}"
        if curvature_key in knowledge_base[line_name]:
            line_reading['Curvature'] = knowledge_base[line_name][curvature_key]

        display_name = line_name.replace('_', ' ').title()
        reading[display_name] = line_reading if line_reading else {'Info': 'No specific interpretation found for the detected features.'}

    return reading


def _draw_lines_on_image(image, lines, original_filename):
    """
    Draws the detected lines on the image and saves it to the static folder.
    Returns the file path of the new image.
    """
    colors = {
        "heart_line": (0, 0, 255), "head_line": (0, 255, 0), "life_line": (255, 0, 0)
    }
    for line_name, line_data in lines.items():
        if line_data['status'] == 'success':
            points = np.array([list(p.values()) for p in line_data['points']], dtype=np.int32)
            cv2.polylines(image, [points], isClosed=False, color=colors.get(line_name, (255,255,255)), thickness=2)

    processed_filename = f"processed_{original_filename}"
    save_path = os.path.join('static', 'processed', processed_filename)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)
    return save_path


def analyze_palm(image_path, original_filename):
    """
    Orchestrates the full palm analysis pipeline, returning the reading and the path to the processed image.
    """
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image = cv2.imread(image_path)
        if image is None:
            return {'status': 'error', 'message': 'Image not found or could not be read.'}

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks_list = [{'index': idx, 'x': lm.x, 'y': lm.y, 'z': lm.z} for idx, lm in enumerate(hand_landmarks.landmark)]

            palm_lines = _find_palm_lines(image.copy(), landmarks_list) # Use a copy for drawing
            line_features = _extract_line_features(palm_lines, (w, h))
            final_reading = _map_features_to_reading(line_features)
            processed_image_path = _draw_lines_on_image(image, palm_lines, original_filename)

            return {
                'status': 'success',
                'reading': final_reading,
                'processed_image_path': processed_image_path
            }
        else:
            return {'status': 'error', 'message': 'No hand detected in the image.'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files or request.files['image'].filename == '':
        return redirect(url_for('index'))

    file = request.files['image']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    analysis_results = analyze_palm(filepath, filename)

    if analysis_results['status'] == 'error':
        return render_template('results.html', error=analysis_results['message'])

    # Generate the URL for the processed image inside the view function
    processed_image_path = analysis_results['processed_image_path']
    # We need to get the filename part from the path to pass to url_for
    processed_filename = os.path.basename(processed_image_path)
    image_url = url_for('static', filename=f'processed/{processed_filename}')

    return render_template('results.html',
                           results=analysis_results['reading'],
                           image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
