from flask import Flask, request, jsonify, send_from_directory, abort
import os
import re
from werkzeug.utils import secure_filename
from pathlib import Path
import argparse

# Define upload folder
DEFAULT_UPLOAD_FOLDER = 'uploads'
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

# Argument parser
p = argparse.ArgumentParser()
p.add_argument('--host', default='0.0.0.0')
p.add_argument('--port', type=int, default=8000)
p.add_argument('--upload-folder', default=DEFAULT_UPLOAD_FOLDER, help='Folder to store uploaded images')
args = p.parse_args()

UPLOAD_FOLDER = args.upload_folder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def get_next_index():
    files = os.listdir(UPLOAD_FOLDER)
    nums = []
    for f in files:
        m = re.match(r"CPE_OPH_(\d+)\.png", f)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'no filename'}), 400
    if f and allowed_file(f.filename):
        idx = get_next_index()
        filename = f"CPE_OPH_{idx}.png"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(save_path)
        return jsonify({'status': 'ok', 'filename': filename})
    return jsonify({'error': 'bad file'}), 400


@app.route('/images', methods=['GET'])
def list_images():
    files = sorted(os.listdir(app.config['UPLOAD_FOLDER']), key=natural_sort_key)
    return jsonify({'files': files})


@app.route('/image/<path:filename>', methods=['GET'])
def serve_image(filename):
    directory = os.path.abspath(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(os.path.join(directory, filename)):
        abort(404)
    return send_from_directory(directory, filename)


@app.route('/download/<path:filename>', methods=['GET'])
def download_image(filename):
    directory = os.path.abspath(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(os.path.join(directory, filename)):
        abort(404)
    return send_from_directory(directory, filename, as_attachment=True)


@app.route('/latest', methods=['GET'])
def latest():
    files = sorted(os.listdir(app.config['UPLOAD_FOLDER']), key=natural_sort_key)
    if not files:
        return jsonify({'latest': None})
    return jsonify({'latest': files[-1]})


if __name__ == '__main__':
    app.run(host=args.host, port=args.port)
