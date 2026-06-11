# DecomposeAI: Diagram Intelligence Engine

DecomposeAI is a modern, full-stack platform designed to perform **zero-shot diagram decomposition** and **editable PowerPoint (.pptx) reconstruction**. It automatically processes educational diagrams (biology cells, circuit diagrams, flowcharts, infographics, textbook scans, and more) using computer vision, segmentations, and OCR, turning flat raster images into layered, editable PowerPoint components.

---

## 🚀 Key Features

* **Zero-Shot Decomposition Pipeline**:
  * **Classification**: Automatic diagram categorization (Biology, Physics, Flowcharts, textbook scans) using OpenCV image heuristics.
  * **Element Detection**: Identifies labels, shapes, arrows, and key boundaries.
  * **High-Fidelity Segmentation**: Extracts pixel-accurate crops and transparent masks for all diagram components.
  * **Text OCR**: Highly precise text extraction associated with bounding boxes.
  * **Structural Understanding**: Establishes parent-child object relationships and labels-to-shape bindings.
  * **Native PowerPoint Compilation**: Reconstructs components directly into native slide elements, text boxes, and layer stacks.
* **Interactive Slide Canvas Dashboard**:
  * Real-time drag-and-drop bounding box editing.
  * Interactive layer list to customize label text, toggle visibility, and control Z-index.
  * Relationship Graph overlay to visualize connected components and annotations.
  * Performance & latency metrics dashboard for tracking pipeline speed.
  * Live-streaming pipeline console logs.

---

## 🛠️ Tech Stack

### Backend
* **FastAPI / Uvicorn**: Lightweight, asynchronous REST API.
* **OpenCV & Pillow**: Main computer vision preprocessing, dimension discovery, and background rendering.
* **SQLAlchemy**: Local SQLite storage for job status, component database rows, and relations.
* **python-pptx**: Native PowerPoint generation and slide building.
* **Celery & Redis**: Background job queue configurations (for production workloads).

### Frontend
* **Vanilla HTML5, CSS3, & Modern JavaScript (ES6)**.
* **Outfit Google Font & FontAwesome Icons** for a premium developer aesthetic.
* **Custom Interactive CSS Canvas** for precise bounding box manipulation.

---

## 📂 Project Structure

```
PPT Generation application/
├── backend/                   # FastAPI Backend
│   ├── pipeline/              # Diagram Decomposition Pipeline
│   │   ├── detectors/         # GroundingDINO Detection Engine
│   │   ├── segmenters/        # SAM2 Shape Segmentation Engine
│   │   ├── ocr/               # PaddleOCR Text Extraction Engine
│   │   ├── classifier.py      # Diagram Classifier
│   │   ├── ppt_generator.py   # PowerPoint (.pptx) Generator
│   │   ├── queue_manager.py   # Job & Task Worker Queue
│   │   └── understanding.py   # Diagram Relationships Resolver
│   ├── database.py            # SQLite/Postgres DB Connection
│   ├── models.py              # SQLAlchemy DB Schema Models
│   ├── requirements.txt       # Python Dependencies
│   └── main.py                # FastAPI Main Server Entrypoint
├── frontend/                  # Interactive Dashboard UI
│   ├── index.html             # Dashboard UI Shell
│   ├── style.css              # Premium Dark-Theme Styling
│   └── app.js                 # Interactive Canvas Logic & API Client
├── test_data/                 # Sample Educational Diagrams & Assets
├── test_dry_run.py            # CLI Script for Local Pipeline Dry Runs
└── .gitignore                 # Excluded directories (databases, caches, temp files)
```

---

## ⚙️ Getting Started

### Prerequisites
* Python 3.9+
* pip

### 1. Backend Setup
1. Navigate to the project root directory:
   ```bash
   cd "PPT Generation application"
   ```
2. Install python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

### 2. Running the Server
Start the Uvicorn application server locally:
```bash
uvicorn backend.main:app --reload --port 8000
```
* The API will be active at `http://127.0.0.1:8000/api`
* The interactive frontend dashboard is served automatically at `http://127.0.0.1:8000/`

### 3. Pipeline Dry Run (Testing)
You can run a pipeline check on sample diagrams without initiating the full server:
```bash
python test_dry_run.py
```
This runs the full classifier, detector, segmenter, OCR, and PowerPoint compiler on sample inputs and saves the results in the `dry_run_output/` folder and root folder.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
