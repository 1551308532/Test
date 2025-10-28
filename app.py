import base64
import io
import os
import re
import uuid
from collections import defaultdict
from statistics import mean, median, pvariance

import easyocr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

UPLOAD_DIR = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.secret_key = 'ocr-analytics-secret-key'

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    return _ocr_reader


def _compute_center(bbox):
    x = sum(point[0] for point in bbox) / 4
    y = sum(point[1] for point in bbox) / 4
    return x, y


def _cluster_by_axis(items, axis_key, threshold):
    clusters = []
    for item in sorted(items, key=lambda value: value[axis_key]):
        if clusters and abs(item[axis_key] - clusters[-1]['avg']) <= threshold:
            clusters[-1]['items'].append(item)
            clusters[-1]['avg'] = np.mean([val[axis_key] for val in clusters[-1]['items']])
        else:
            clusters.append({'items': [item], 'avg': item[axis_key]})
    return clusters


def _merge_header_candidates(candidates):
    if not candidates:
        return []
    sorted_candidates = sorted(candidates, key=lambda item: item['x'])
    merged = []
    current = {'texts': [sorted_candidates[0]['text']], 'xs': [sorted_candidates[0]['x']]}
    for candidate in sorted_candidates[1:]:
        if candidate['x'] - np.mean(current['xs']) <= 60:
            current['texts'].append(candidate['text'])
            current['xs'].append(candidate['x'])
        else:
            merged.append({
                'name': re.sub(r'\s+', ' ', ' '.join(current['texts'])).strip(),
                'x': float(np.mean(current['xs']))
            })
            current = {'texts': [candidate['text']], 'xs': [candidate['x']]}
    merged.append({
        'name': re.sub(r'\s+', ' ', ' '.join(current['texts'])).strip(),
        'x': float(np.mean(current['xs']))
    })
    return merged


def _extract_table_entries(ocr_results):
    processed = []
    max_y = 0
    for bbox, text, confidence in ocr_results:
        cleaned = text.strip()
        if not cleaned:
            continue
        x, y = _compute_center(bbox)
        max_y = max(max_y, y)
        processed.append({'text': cleaned, 'x': float(x), 'y': float(y)})

    if not processed:
        return [], [], []

    header_threshold = max_y * 0.22
    header_candidates = [item for item in processed if item['y'] <= header_threshold]
    headers = _merge_header_candidates(header_candidates)
    if not headers:
        return [], [], []

    data_items = [item for item in processed if item['y'] > header_threshold]
    for item in data_items:
        column_index = min(range(len(headers)), key=lambda idx: abs(item['x'] - headers[idx]['x']))
        item['column'] = column_index

    experiment_items = [item for item in data_items if item['column'] == 0]
    experiment_clusters = _cluster_by_axis(experiment_items, 'y', threshold=45)
    experiments = []
    for cluster in experiment_clusters:
        ordered_texts = sorted(cluster['items'], key=lambda item: item['x'])
        name = re.sub(r'\s+', ' ', ' '.join(value['text'] for value in ordered_texts)).strip()
        if name:
            experiments.append({'name': name, 'y': float(cluster['avg'])})
    experiments = sorted(experiments, key=lambda entry: entry['y'])

    experiment_order = [exp['name'] for exp in experiments]
    if not experiment_order:
        return [], [], []

    cell_map = defaultdict(list)
    for item in data_items:
        if item['column'] == 0:
            continue
        target_index = None
        min_distance = 9999
        for idx, exp in enumerate(experiments):
            distance = abs(item['y'] - exp['y'])
            if distance < min_distance and distance <= 55:
                min_distance = distance
                target_index = idx
        if target_index is not None:
            cell_map[(target_index, item['column'])].append(item)

    entries = []
    salt_pattern = re.compile(r"盐值[:=：]?\s*([-+]?\d+(?:\.\d+)?)%?[^-+]*?z值[:=：]?\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)

    for (exp_idx, col_idx), items in cell_map.items():
        header_idx = min(col_idx, len(headers) - 1)
        parameter = re.sub(r'\s+', ' ', headers[header_idx]['name']).strip()
        if not parameter:
            continue
        ordered_items = sorted(items, key=lambda entry: (entry['y'], entry['x']))
        cell_text = re.sub(r'\s+', ' ', ' '.join(value['text'] for value in ordered_items))
        for salt_value, z_value in salt_pattern.findall(cell_text):
            try:
                salt = float(salt_value)
                z = float(z_value)
            except ValueError:
                continue
            entries.append({
                'experiment': experiments[exp_idx]['name'],
                'parameter': parameter,
                'salt': salt,
                'z': z
            })

    return entries, experiment_order, [header['name'] for header in headers]


def _compute_salt_statistics(entries):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry['salt']].append(abs(entry['z']))

    salt_stats = []
    for salt in sorted(grouped.keys()):
        values = grouped[salt]
        total = float(np.sum(values))
        avg = float(mean(values)) if values else 0.0
        med = float(median(values)) if values else 0.0
        var = float(pvariance(values)) if len(values) > 1 else 0.0
        salt_stats.append({
            'salt': salt,
            'count': len(values),
            'sum_abs': total,
            'mean_abs': avg,
            'median_abs': med,
            'variance_abs': var
        })
    return salt_stats


def _generate_parameter_charts(entries, experiments):
    parameter_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for entry in entries:
        parameter_map[entry['parameter']][entry['salt']][entry['experiment']].append(entry['z'])

    colors = plt.cm.get_cmap('tab10')
    charts = []
    for idx, (parameter, salt_groups) in enumerate(parameter_map.items()):
        fig, ax = plt.subplots(figsize=(8, 4))
        sorted_salts = sorted(salt_groups.keys())
        for color_idx, salt in enumerate(sorted_salts):
            series = []
            for experiment in experiments:
                values = salt_groups[salt].get(experiment, [])
                if values:
                    series.append(float(np.mean(values)))
                else:
                    series.append(np.nan)
            ax.plot(experiments, series, marker='o', label=f'盐值 {salt}', color=colors(color_idx % colors.N))
        ax.set_title(f'{parameter} - Z值趋势')
        ax.set_xlabel('实验组')
        ax.set_ylabel('Z值')
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        charts.append({'parameter': parameter, 'image': encoded})
        plt.close(fig)
        if len(charts) >= 5:
            break
    return charts


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        flash('请先上传图片。')
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        flash('请选择要上传的图片。')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}" if filename else f"{uuid.uuid4().hex}.png"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(file_path)

    try:
        try:
            Image.open(file_path)
        except Exception:
            flash('无法打开上传的图片，请确认文件格式正确。')
            return redirect(url_for('index'))

        reader = get_ocr_reader()
        ocr_results = reader.readtext(file_path, detail=1)
        entries, experiment_order, headers = _extract_table_entries(ocr_results)

        if not entries:
            flash('未能从图片中识别到有效的盐值和Z值数据。请尝试更清晰的表格图片。')
            return redirect(url_for('index'))

        salt_stats = _compute_salt_statistics(entries)
        charts = _generate_parameter_charts(entries, experiment_order)

        return render_template(
            'results.html',
            entries=entries,
            experiments=experiment_order,
            salt_stats=salt_stats,
            charts=charts,
            headers=headers
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == '__main__':
    app.run(debug=True)
