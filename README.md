# IndoDisaster-MLOps

## Prediksi Status Keadaan Darurat Bencana di Indonesia Menggunakan Machine Learning Berdasarkan Data Kejadian Bencana Tahun 2025

**IndoDisaster-MLOps** adalah project Machine Learning dan MLOps untuk memprediksi **status keadaan darurat bencana di Indonesia** berdasarkan karakteristik kejadian bencana dan informasi geografis yang tersedia.

Project ini tidak hanya berfokus pada pembangunan model Machine Learning, tetapi juga mengimplementasikan proses deployment dan monitoring model melalui **FastAPI, Docker, Prometheus, dan Grafana**.

---

## 🎯 Project Objective

Tujuan utama project ini adalah membangun model klasifikasi yang dapat memprediksi status keadaan darurat suatu kejadian bencana berdasarkan informasi yang tersedia pada awal kejadian atau saat penetapan status.

Model memprediksi tiga kelas:

* **Siaga Darurat**
* **Tanggap Darurat**
* **Transisi Darurat ke Pemulihan**

Project ini juga dirancang sebagai implementasi praktik **MLOps**, mulai dari pengembangan model hingga penyediaan API dan monitoring service.

---

## 🧠 Machine Learning

### Problem Type

**Multiclass Classification**

### Target

`status_keadaan_darurat`

### Classes

| Class                         | Description                                     |
| ----------------------------- | ----------------------------------------------- |
| Siaga Darurat                 | Status keadaan darurat pada tahap kesiapsiagaan |
| Tanggap Darurat               | Status pada tahap penanganan keadaan darurat    |
| Transisi Darurat ke Pemulihan | Status pada tahap peralihan menuju pemulihan    |

---

## 📊 Dataset

Project menggunakan data kejadian dan penetapan status keadaan darurat bencana di Indonesia tahun **2025**.

### Dataset 1 — Data Mikro Status Keadaan Darurat Bencana

Dataset utama berisi informasi mengenai:

* Jenis bencana
* Klasifikasi bencana
* Skala bencana
* Provinsi
* Kabupaten
* Status keadaan darurat
* Tanggal mulai
* Informasi geografis

Dataset utama memiliki **495 data kejadian**.

### Dataset 2 — Kabupaten yang Menetapkan Status Keadaan Darurat Tahun 2025

Dataset tambahan digunakan untuk memperoleh informasi geografis kabupaten/kota, seperti:

* Latitude
* Longitude
* ID Kabupaten

Dataset ini memiliki **217 wilayah kabupaten/kota**.

> Kolom yang secara langsung mengungkap target atau berpotensi menyebabkan data leakage tidak digunakan sebagai fitur model.

---

## 🔎 Exploratory Data Analysis

Tahap Exploratory Data Analysis (EDA) dilakukan untuk memahami karakteristik dataset sebelum proses pemodelan.

Analisis meliputi:

* Distribusi status keadaan darurat
* Distribusi jenis bencana
* Distribusi klasifikasi bencana
* Distribusi kejadian berdasarkan provinsi
* Distribusi kejadian berdasarkan bulan
* Analisis missing values
* Analisis data geografis
* Pemeriksaan duplikasi data
* Pemeriksaan konsistensi tanggal

### Distribusi Target

| Status                        |  Jumlah |
| ----------------------------- | ------: |
| Tanggap Darurat               |     287 |
| Siaga Darurat                 |     163 |
| Transisi Darurat ke Pemulihan |      45 |
| **Total**                     | **495** |

Dataset memiliki ketidakseimbangan kelas, terutama pada kelas **Transisi Darurat ke Pemulihan**.

---

## ⚙️ Data Processing & Feature Engineering

Tahapan preprocessing dan feature engineering dilakukan untuk menghasilkan fitur yang dapat digunakan oleh model.

Beberapa fitur yang digunakan:

### Categorical Features

* `jenis_bencana`
* `klasifikasi_bencana`
* `skala_bencana`
* `provinsi`
* `kabupaten`

### Numerical Features

* `latitude`
* `longitude`
* `bulan_mulai`
* `kuartal_mulai`
* `jumlah_kejadian_sebelumnya`
* `hari_sejak_kejadian_sebelumnya`
* `ada_kejadian_sebelumnya`

### Preprocessing

Categorical features diproses menggunakan:

* Missing value imputation
* One-Hot Encoding

Numerical features diproses menggunakan:

* Median imputation
* StandardScaler

Seluruh preprocessing dimasukkan ke dalam **Scikit-learn Pipeline** untuk mencegah data leakage antara training dan testing.

---

## 🤖 Model Development

Beberapa algoritma Machine Learning dibandingkan:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Evaluasi dilakukan menggunakan **5-Fold Stratified Cross Validation** dan pengujian menggunakan data test.

Metric yang digunakan:

* Accuracy
* Precision Macro
* Recall Macro
* F1-Score Macro

---

## 📈 Model Evaluation

Hasil evaluasi pada data testing:

| Model               |   Accuracy | Precision Macro | Recall Macro |   F1 Macro |
| ------------------- | ---------: | --------------: | -----------: | ---------: |
| Logistic Regression |     75.76% |          67.88% |   **73.63%** | **69.81%** |
| Decision Tree       |     73.74% |      **69.09%** |       62.25% |     64.38% |
| Random Forest       | **78.79%** |          52.71% |       57.52% |     55.00% |
| Gradient Boosting   |     75.76% |          52.59% |       54.07% |     53.01% |

Random Forest memperoleh accuracy tertinggi pada test set, tetapi tidak mendeteksi kelas **Transisi Darurat ke Pemulihan** pada data testing.

Logistic Regression menghasilkan **Macro F1 tertinggi** dan mampu mendeteksi ketiga kelas target. Oleh karena itu, **Logistic Regression digunakan sebagai model final untuk tahap deployment dan MLOps**.

### Final Model Performance

* **Accuracy:** 75.76%
* **Precision Macro:** 67.88%
* **Recall Macro:** 73.63%
* **F1 Macro:** 69.81%

Model final disimpan menggunakan format:

```text
models/final_model.joblib
```

---

# 🚀 MLOps Implementation

Setelah model selesai dikembangkan, project dilanjutkan ke tahap MLOps untuk menyediakan model sebagai service dan melakukan monitoring terhadap API.

Arsitektur project:

```text
                    ┌─────────────────────┐
                    │   Client / User     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │    /predict         │
                    │     /health        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  ML Model Pipeline  │
                    │ Logistic Regression │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Prediction Result   │
                    └─────────────────────┘

          ┌────────────────────────────────────┐
          │          Monitoring Layer          │
          │                                    │
          │  FastAPI → Prometheus → Grafana   │
          └────────────────────────────────────┘
```

---

## ⚡ FastAPI

Model disediakan melalui REST API menggunakan **FastAPI**.

Endpoint yang tersedia:

### `GET /`

Memeriksa apakah API berjalan.

Example response:

```json
{
  "message": "IndoDisaster MLOps API is running",
  "model": "Logistic Regression"
}
```

### `GET /health`

Digunakan untuk health check service.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`

Digunakan untuk melakukan prediksi status keadaan darurat.

Contoh request:

```json
{
  "jenis_bencana": "Banjir",
  "klasifikasi_bencana": "Hidrometeorologi Basah",
  "skala_bencana": "kabupaten",
  "provinsi": "Jawa Barat",
  "kabupaten": "Bandung",
  "latitude": -6.9175,
  "longitude": 107.6191,
  "bulan_mulai": 12,
  "kuartal_mulai": 4,
  "jumlah_kejadian_sebelumnya": 2,
  "hari_sejak_kejadian_sebelumnya": 14,
  "ada_kejadian_sebelumnya": 1
}
```

Example response:

```json
{
  "prediction": "Tanggap Darurat"
}
```

### `GET /metrics`

Endpoint yang digunakan oleh Prometheus untuk mengambil metrics dari API.

---

# 🐳 Docker

API dikemas menggunakan **Docker** sehingga environment aplikasi dan dependencies dapat dijalankan secara konsisten.

Docker image menjalankan:

```text
Python 3.9
FastAPI
Uvicorn
Scikit-learn
Pandas
Joblib
Prometheus FastAPI Instrumentator
```

API dapat dijalankan melalui:

```bash
docker build -t indodisaster-mlops .
```

Kemudian:

```bash
docker run -p 8000:8000 indodisaster-mlops
```

API dapat diakses melalui:

```text
http://localhost:8000
```

---

# 📡 Prometheus

**Prometheus** digunakan sebagai monitoring system untuk mengumpulkan metrics dari FastAPI.

Metrics yang tersedia antara lain:

* Total HTTP requests
* HTTP request duration
* Request count
* Response size
* HTTP status code

Contoh metric:

```promql
http_requests_total
```

Prometheus berjalan pada:

```text
http://localhost:9090
```

Konfigurasi scraping menggunakan service API:

```yaml
scrape_configs:
  - job_name: "indodisaster-api"
    static_configs:
      - targets: ["api:8000"]
```

---

# 📊 Grafana

**Grafana** digunakan untuk melakukan visualisasi metrics dari Prometheus dalam bentuk dashboard monitoring.

Dashboard dirancang untuk memonitor:

* Total API Requests
* API Response Time
* Request Rate
* Error Rate
* Prediction Distribution

Grafana berjalan pada:

```text
http://localhost:3000
```

Data source yang digunakan:

```text
Prometheus
```

---

# 🧩 Docker Compose

Seluruh service MLOps dijalankan menggunakan Docker Compose.

Service yang digunakan:

```text
┌─────────────────────────────┐
│        Docker Compose       │
│                             │
│  ┌───────┐  ┌────────────┐ │
│  │ FastAPI│  │ Prometheus│ │
│  │ :8000 │  │   :9090   │ │
│  └───┬───┘  └─────┬──────┘ │
│      │             │        │
│      │             ▼        │
│      │        ┌─────────┐  │
│      └───────►│ Grafana │  │
│               │  :3000  │  │
│               └─────────┘  │
└─────────────────────────────┘
```

Untuk menjalankan seluruh service:

```bash
docker compose up --build
```

Untuk menjalankan secara background:

```bash
docker compose up -d
```

Melihat status container:

```bash
docker compose ps
```

---

# 📁 Project Structure

```text
IndoDisaster-MLOps/
│
├── app/
│   └── main.py
│
├── models/
│   └── final_model.joblib
│
├── prometheus/
│   └── prometheus.yml
│
├── notebooks/
│   └── ...
│
├── data/
│   └── ...
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 🛠️ Tech Stack

### Programming & Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

### API

* FastAPI
* Uvicorn
* Pydantic

### MLOps & Deployment

* Docker
* Docker Compose
* Prometheus
* Grafana

### Development

* Jupyter Notebook
* Git
* GitHub

---

# 🔄 Project Workflow

```text
Data Collection
      ↓
Exploratory Data Analysis
      ↓
Data Cleaning
      ↓
Data Integration
      ↓
Feature Engineering
      ↓
Feature Selection
      ↓
Train-Test Split
      ↓
Model Development
      ↓
Model Evaluation
      ↓
Final Model
      ↓
Model Serialization
      ↓
FastAPI
      ↓
Docker
      ↓
Prometheus
      ↓
Grafana
      ↓
Monitoring
```

---

# 🎯 Future Development

Beberapa pengembangan yang dapat dilakukan selanjutnya:

* Implementasi GitHub Actions untuk CI/CD
* Automated testing untuk API
* Automated Docker image build
* Prediction monitoring
* Monitoring distribution setiap kelas prediksi
* Model performance monitoring
* Model retraining pipeline
* Model versioning
* Deployment ke cloud infrastructure
* Implementasi alerting pada Prometheus/Grafana

---

# 👨‍💻 Author

**Margohan L. Siringo-Ringo**

Informatics | Machine Learning | Artificial Intelligence | MLOps

Interested in:

* AI/ML Engineering
* Computer Vision
* Machine Learning
* Data Science
* MLOps

---

## ⭐ Project Purpose

Project ini dibuat sebagai **hands-on learning project** untuk memahami bagaimana sebuah model Machine Learning dikembangkan dan kemudian dibawa ke lingkungan yang lebih menyerupai production melalui API, containerization, monitoring, dan automation.

> **From Machine Learning Model to Monitored MLOps Service.**
