import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # ✅ Tambahkan untuk mengatasi masalah CORS
import joblib

app = Flask(__name__)
CORS(app)  # ✅ Aktifkan CORS agar bisa diakses dari domain lain (frontend di hosting berbeda)

# Lokasi model dan label encoder
MODEL_DIR = '.'
IMAGE_SIZE = (64, 64)

# Load model dan label encoder
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'model_rf.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    print("✅ Model dan Label Encoder berhasil dimuat.")
except FileNotFoundError:
    print("❌ Error: model_rf.pkl atau label_encoder.pkl tidak ditemukan di direktori.")
    model = None
    le = None
except Exception as e:
    print(f"❌ Gagal memuat model atau encoder: {e}")
    model = None
    le = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or le is None:
        return jsonify({'error': 'Model atau Label Encoder belum tersedia.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'File tidak ditemukan dalam permintaan.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File kosong.'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img).flatten() / 255.0
        img_array = img_array.reshape(1, -1)

        prediction_encoded = model.predict(img_array)
        prediction_label = le.inverse_transform(prediction_encoded)[0]

        probabilities = model.predict_proba(img_array)[0]
        confidence = np.max(probabilities) * 100

        return jsonify({
            'prediction': prediction_label,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        print(f"❌ Error saat memproses gambar: {e}")
        return jsonify({'error': f'Kesalahan saat memproses gambar: {str(e)}'}), 500

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Terjadi kesalahan server (500).'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Halaman tidak ditemukan (404).'}), 404

if __name__ == '__main__':
    # Pastikan direktori dan file HTML tersedia
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("📁 Folder 'templates' dibuat.")

    if os.path.exists('index.html') and not os.path.exists('templates/index.html'):
        os.rename('index.html', 'templates/index.html')
        print("📄 index.html dipindahkan ke folder 'templates'.")

    elif not os.path.exists('templates/index.html'):
        print("⚠️ Warning: File 'index.html' belum ditemukan.")

    app.run(debug=True, host='0.0.0.0', port=5000)
