# Cloudera Image Analysis

Extract, transcribe, and analyze content from document images using **local vision AI powered by Ollama** — running entirely on-cluster with no external API dependencies.

The application uses **Qwen2.5-VL**, the leading open-source model for document understanding and OCR (864/1000 on OCRBench, 95.7% on DocVQA), served locally via Ollama on your CML GPU allocation.

---

## Use Cases

| Use Case | What It Does |
|---|---|
| **Transcribing Typed Text** | Verbatim OCR — outputs every character exactly as printed, preserving line breaks and punctuation |
| **Transcribing Handwritten Text** | Digitises handwritten notes character-by-character; marks illegible words as `[illegible]` |
| **Transcribing Forms** | Extracts every field label and value as structured `Label: Value` pairs |
| **Complicated Document QA** | Answers a specific question grounded in the document content |
| **Unstructured → JSON** | Converts document content to well-structured, snake_case JSON |
| **Summarize Image** | Describes the image content, purpose, and key takeaways in clear prose |

---

## Architecture

```
Image (PNG / JPG / WEBP / GIF)
        │
        ▼
  prepare_image()          ← resize to 1280px max, JPEG quality 92
        │
        ▼
  Ollama /api/chat         ← qwen2.5vl:7b (default) running on local GPU
        │   JSON Schema format constraint per use case
        │   (grammar-enforced structured output)
        ▼
  _parse_structured_response()  ← renders schema output to display text
        │
        ▼
  Streaming SSE  ──────────────────────────▶  Browser (token-by-token)
```

### Why Qwen2.5-VL?

| Model | OCRBench | DocVQA | VRAM |
|---|---|---|---|
| **Qwen2.5-VL 7B** (default) | **864 / 1000** | **95.7%** | ~6 GB |
| Qwen2.5-VL 32B (optional) | higher | **96.4%** | ~21 GB |
| Llama 3.2 Vision 11B (legacy) | ~600 | ~80% | ~8 GB |

Qwen2.5-VL is purpose-built for document understanding with dynamic-resolution image encoding (28×28 patches), 125K context window, and native structured output support.

### Output guardrails

Structured use cases (Forms, QA, Summarize) pass a **JSON Schema** as Ollama's `format` parameter — enforced at the token-sampling level via GBNF grammars. The model cannot produce text outside the schema, eliminating hallucinated commentary without post-hoc filtering.

Transcription use cases (Typed Text, Handwritten Text) use plain-text streaming with aggressive sampling parameters (`temperature: 0.05`, `repeat_penalty: 1.4`, `num_ctx: 65536`) to handle arbitrary content including code, quotes, and special characters verbatim.

---

## Deployment on Cloudera Machine Learning (CML)

This is a CML **Accelerator for ML Projects (AMP)**. Deploy from the CML catalog or clone this repository directly as a CML project.

### Automated setup (`.project-metadata.yaml`)

| Step | Script | What it does |
|---|---|---|
| 1 | `1_session-install-dependencies/install.py` | Installs Python dependencies |
| 2 | `2_setup_models/setup_models.py` | Installs Ollama (no root), pulls `qwen2.5vl:7b` |
| 3 | `3_application/start-app.py` | Launches FastAPI + custom HTML/CSS/JS UI |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOCAL_MODEL` | `qwen2.5vl:7b` | Ollama model to pull and use. Options: `qwen2.5vl:7b`, `qwen2.5vl:32b`, `llama3.2-vision:11b` |

### Resource requirements

| Resource | Recommended |
|---|---|
| GPU | 1 × NVIDIA L4 (24 GB) or equivalent |
| VRAM | 6 GB minimum (qwen2.5vl:7b), 21 GB for 32b |
| CPU | 4 cores |
| RAM | 16 GB |

---

## Features

### Single Image Analysis
Select a use case, choose an uploaded or example image, and click **Process Image**. Results stream token-by-token. A **Stop** button cancels in-flight generation.

### Batch Processing
Select multiple images and a use case, then submit. Jobs run in a background queue and results are saved as `.txt` files for download.

### Folder Management
Create named folders, upload images directly into them, and download all results for a folder as a `.zip` file.

### Personalization
The UI reads CML environment variables (`GIT_AUTHOR_NAME`, `PROJECT_OWNER`, `CDSW_PROJECT`, `CDSW_DOMAIN`) to display your name, project, and a personalized greeting.

### Configuration Tab
Change the active model, view GPU status and VRAM usage, and pull new models from the Ollama registry — all from the UI.

---

## Local Development

```bash
# Install dependencies
pip install -r 1_session-install-dependencies/requirements.txt

# Install Ollama and pull the model
python 2_setup_models/setup_models.py

# Run the app
python 3_application/start-app.py
```

The app listens on `$CDSW_APP_PORT` (default `8090`).

---

## Project Structure

```
.
├── 1_session-install-dependencies/
│   ├── requirements.txt          # Python dependencies
│   └── install.py                # CML job: pip install
├── 2_setup_models/
│   └── setup_models.py           # CML job: installs Ollama + pulls model
├── 3_application/
│   ├── app.py                    # FastAPI backend
│   ├── start-app.py              # CML application entry point
│   └── static/
│       ├── index.html            # Single-page UI
│       ├── styles.css            # Cloudera Workbench design system
│       └── app.js                # Frontend logic
├── data/
│   └── examples/                 # Bundled example images
│       ├── ex1-stack_overflow.png
│       ├── ex2-school_notes.png
│       ├── ex3-vehicle_form.jpeg
│       ├── ex4-doc_qa.jpeg
│       └── ex5-org_chart.jpeg
├── .project-metadata.yaml        # CML AMP deployment specification
├── catalog-entry.yaml            # Cloudera catalog metadata
└── README.md
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
