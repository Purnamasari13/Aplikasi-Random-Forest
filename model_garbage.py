# 1. Import library
import os
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# 2. Muat dataset gambar
IMAGE_SIZE = (64, 64)
DATASET_DIR = 'Garbage classification'

data = []
labels = []

print("🔄 Memuat gambar...")
for label_name in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label_name)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(label_path, file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(IMAGE_SIZE)
                    data.append(np.array(img).flatten())
                    labels.append(label_name)
                except Exception as e:
                    print(f"❌ Gagal memuat {img_path}: {e}")
                    continue

# 3. Eksplorasi Dataset: Distribusi label
print("\n📊 Jumlah gambar per kelas:")
label_counts = Counter(labels)
for k, v in label_counts.items():
    print(f"{k}: {v} gambar")

# Visualisasi distribusi label
df_counts = pd.DataFrame.from_dict(label_counts, orient='index', columns=['Jumlah'])
df_counts.plot(kind='bar', title='Distribusi Gambar per Kelas', ylabel='Jumlah', xlabel='Kelas')
plt.tight_layout()
plt.show()

# Visualisasi contoh gambar
print("\n🖼️ Menampilkan contoh gambar per kelas...")
fig, axes = plt.subplots(1, len(label_counts), figsize=(15, 5))
for ax, class_name in zip(axes, label_counts.keys()):
    folder = os.path.join(DATASET_DIR, class_name)
    sample_image = next((f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))), None)
    if sample_image:
        img = Image.open(os.path.join(folder, sample_image)).resize(IMAGE_SIZE)
        ax.imshow(img)
        ax.set_title(class_name)
        ax.axis('off')
plt.suptitle("Contoh Gambar per Kelas")
plt.tight_layout()
plt.show()

# 4. Preprocessing Data
X = np.array(data) / 255.0  # Normalisasi fitur (pixel 0–255 → 0–1)
le = LabelEncoder()
y = le.fit_transform(labels)

# 5. Pisahkan data latih dan uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Buat model Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)

# 7. Latih model
model.fit(X_train, y_train)

# 8. Prediksi data uji
y_pred = model.predict(X_test)

# 9. Evaluasi Model
print("\n🎯 Akurasi Model:", accuracy_score(y_test, y_pred))
print("\n📄 Laporan Klasifikasi:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 10. Tampilkan Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Prediksi")
plt.ylabel("Sebenarnya")
plt.title("Confusion Matrix - Klasifikasi Sampah")
plt.tight_layout()
plt.show()

# 11. Analisis Fitur Penting
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Pixel": [f"pixel_{i}" for i in range(len(feature_importances))],
    "Importance": feature_importances
})

importance_df = importance_df.sort_values(by="Importance", ascending=False).head(20)

# 12. Plot fitur yang paling berpengaruh
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Pixel", data=importance_df, palette="viridis")
plt.xlabel("Tingkat Kepentingan")
plt.ylabel("Fitur (Pixel)")
plt.title("20 Pixel Paling Berpengaruh pada Random Forest")
plt.tight_layout()
plt.show()

# 13. Simpan model dan label encoder
joblib.dump(model, "model_rf.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\n✅ Model dan label encoder berhasil disimpan.")
