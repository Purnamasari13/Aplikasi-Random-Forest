import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import joblib
import io

app = Flask(__name__)

# Direktori tempat model dan label encoder disimpan
MODEL_DIR = '.' # Asumsi model_rf.pkl dan label_encoder.pkl ada di direktori yang sama

# Ukuran gambar yang digunakan saat pelatihan model
IMAGE_SIZE = (64, 64)

# Muat model dan label encoder saat aplikasi dimulai
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'model_rf.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    print("✅ Model dan Label Encoder berhasil dimuat.")
except FileNotFoundError:
    print(f"❌ Error: Pastikan 'model_rf.pkl' dan 'label_encoder.pkl' ada di {MODEL_DIR}")
    model = None
    le = None
except Exception as e:
    print(f"❌ Error saat memuat model/label encoder: {e}")
    model = None
    le = None

@app.route('/')
def index():
    """Render halaman HTML utama."""
    return render_template('index.html') # Pastikan index.html berada di folder 'templates'

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk melakukan prediksi klasifikasi sampah."""
    if model is None or le is None:
        return jsonify({'error': 'Model atau Label Encoder belum dimuat. Periksa log server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada bagian file dalam permintaan.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih.'}), 400

    if file:
        try:
            # Baca gambar dari request
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            img = img.resize(IMAGE_SIZE)
            img_array = np.array(img).flatten() / 255.0 # Normalisasi sama seperti saat pelatihan

            # Reshape untuk satu sampel
            img_array = img_array.reshape(1, -1)

            # Lakukan prediksi
            prediction_encoded = model.predict(img_array)
            prediction_label = le.inverse_transform(prediction_encoded)[0]

            # Dapatkan probabilitas prediksi
            probabilities = model.predict_proba(img_array)[0]
            confidence = np.max(probabilities) * 100 # Konversi ke persentase

            return jsonify({
                'prediction': prediction_label,
                'confidence': float(f"{confidence:.2f}") # Format ke 2 desimal
            })
        except Exception as e:
            return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {e}'}), 500

    return jsonify({'error': 'Terjadi kesalahan tidak dikenal.'}), 500

if __name__ == '__main__':
    # Buat folder 'templates' jika belum ada
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Folder 'templates' dibuat.")

    # Pindahkan index.html ke folder 'templates'
    if os.path.exists('index.html') and not os.path.exists('templates/index.html'):
        os.rename('index.html', 'templates/index.html')
        print("index.html dipindahkan ke folder 'templates'.")
    elif not os.path.exists('index.html') and not os.path.exists('templates/index.html'):
        print("Peringatan: index.html tidak ditemukan. Pastikan Anda telah menyimpan HTML sebelumnya.")


    # Anda bisa mengatur host='0.0.0.0' agar bisa diakses dari perangkat lain di jaringan yang sama
    # Atau biarkan default (localhost) untuk pengembangan lokal
    app.run(debug=True, host='0.0.0.0', port=5000)