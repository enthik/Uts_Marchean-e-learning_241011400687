# UTS Machine Learning – Linear Regression & Logistic Regression

- **Nama** : Valentina Nelu
- **NIM** : 241011400687
- **Kelas** : Reg B

## Isi Repositori

| File | Keterangan |
| --- | --- |
| `tugas_UTS.ipynb` | Notebook jawaban, sudah dijalankan (Run All) lengkap dengan output dan grafik |
| `Ads_Data.csv` | Dataset biaya iklan vs penjualan (40 data) |
| `Email_Data.csv` | Dataset email spam (60 data) |

Model dan metrik evaluasi diimplementasikan sendiri dengan NumPy (`numpy`, `pandas`, `matplotlib`). scikit-learn hanya dipakai pada soal Bonus.

## Ringkasan Hasil (`RANDOM_STATE = 87`)

### Bagian A – Linear Regression (Ads_Data, split 80/20)

- Persamaan (normal equation): **ŷ = 28,1481 + 9,2856 · BiayaIklan**. Setiap kenaikan biaya iklan Rp1 juta, penjualan naik sekitar Rp9,29 juta.

| Data | MAE | MSE | RMSE | R² |
| --- | --- | --- | --- | --- |
| Training | 4,0245 | 27,9056 | 5,2826 | 0,9473 |
| Testing | 5,9217 | 72,8966 | 8,5380 | 0,9196 |

- Gradient descent (5000 iterasi): lr = 0,001 terlalu lambat (b = 23,37; m = 10,04); lr = 0,02 pas dan sama dengan normal equation; lr = 0,1 divergen (overflow, hasil NaN mulai iterasi ke-199).
- Prediksi: 4,5 juta → 69,93 juta; 7 juta → 93,15 juta; 12 juta → 139,57 juta (ekstrapolasi, kurang bisa dipercaya).

### Bagian B – Logistic Regression (Email_Data, split 75/25, lr = 0,1, 10000 iterasi)

- Persamaan: **P(Spam) = σ(−9,2040 + 0,8013 · JumlahLink + 0,8248 · KataPromo)**.
- Evaluasi data testing (15 email): accuracy 0,933; precision 0,833; recall 1,00; F1 0,909; confusion matrix TN = 9, FP = 1, FN = 0, TP = 5.
- Decision boundary: x₂ = 11,1589 − 0,9715 · x₁. Ada 1 email testing yang salah diklasifikasikan (FP).

| Threshold | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- |
| 0,3 | 2 | 0 | 0,714 | 1,00 |
| 0,5 | 1 | 0 | 0,833 | 1,00 |
| 0,7 | 1 | 0 | 0,833 | 1,00 |

- Threshold yang dipilih: **0,7** (FP lebih merugikan pengguna).
- Prediksi email baru (threshold 0,7): E1 Bukan Spam, E2 Bukan Spam, E3 Spam, E4 Spam.

### Bagian C & Bonus

- Jawaban konsep C1–C5 ada di notebook.
- Bonus: scikit-learn memberi hasil identik untuk Linear Regression. Untuk Logistic Regression koefisien berbeda karena scikit-learn memakai regularisasi L2 (`C = 1.0`) secara default dan solver yang berbeda; akurasi testing sama (0,933).
