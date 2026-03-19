# Image OCR & Analysis with Cloudera AI Inference

Transcription and information extraction from document images — powered entirely by **NVIDIA NIM models hosted on Cloudera's AI Inference Service**.

This AMP replicates the familiar image-analysis workflow but replaces external cloud APIs with models you control inside your Cloudera environment. It combines a dedicated document OCR model with a large language model to deliver both accurate text extraction and intelligent, context-aware analysis.

---

## What It Does

| Use Case | Description |
|---|---|
| **Transcribing Typed Text** | Extract printed text from scanned PDFs, screenshots, or documents |
| **Transcribing Handwritten Text** | Digitise handwritten notes |
| **Transcribing Forms** | Extract field labels and values from structured forms |
| **Document QA** | Answer a specific question grounded in the document |
| **Unstructured → JSON** | Convert document content to well-structured JSON |
| **User Defined** | Write a fully custom prompt |

---

## Architecture

The application supports three processing modes:

```
OCR → LLM Pipeline  (default, recommended)
┌─────────┐    ┌──────────────────────┐    ┌──────────────────┐
│  Image  │───▶│  NeMo Retriever-Parse│───▶│ Llama 3.3 70B /  │───▶ Result
│         │    │  (or PaddleOCR)      │    │ other LLM        │
└─────────┘    └──────────────────────┘    └──────────────────┘

Vision LLM mode
┌─────────┐    ┌──────────────────────┐
│  Image  │───▶│  Multimodal LLM      │───▶ Result
└─────────┘    └──────────────────────┘

OCR Only
┌─────────┐    ┌──────────────────────┐
│  Image  │───▶│  NeMo Retriever-Parse│───▶ Extracted text
└─────────┘    └──────────────────────┘
```

### Models Used

| Stage | Model | Why |
|---|---|---|
| **OCR / Text Extraction** | **NeMo Retriever-Parse** | Purpose-built for document images; extracts formatted text with semantic labels (headings, paragraphs, tables) |
| **Analysis / Reasoning** | **Llama 3.3 70B Instruct** | The strongest general-purpose model in the Cloudera AI Inference catalog; handles all use cases accurately |

Both models run entirely within your Cloudera environment — no external API calls.

---

## Deployment on Cloudera Machine Learning (CML)

This is a CML Accelerator for ML Projects (AMP). Deploy it directly from the CML AMP catalog or by cloning this repository as a CML project.

### Automated deployment tasks

The `.project-metadata.yaml` defines:
1. **Install Dependencies** — runs `1_session-install-dependencies/install.py`
2. **Start Application** — runs `2_application/start-app.py` (Streamlit on `$CDSW_APP_PORT`)

### Environment variables (optional)

You can pre-configure the application via CML environment variables, or use the **⚙️ Configuration** tab in the UI:

| Variable | Description |
|---|---|
| `CAI_INFERENCE_TOKEN` | CDP JWT token (used for both endpoints) |
| `CAI_OCR_ENDPOINT_URL` | Full URL of the NeMo Retriever-Parse endpoint |
| `CAI_LLM_ENDPOINT_URL` | Full URL of the Llama 3.3 70B endpoint |

### Endpoint URL format

```
https://ml-<workspace-id>.<env>.<tenant>.cloudera.site/namespaces/serving-default/endpoints/<endpoint-name>/v1
```

Example:
```
https://ml-64288d82-5dd.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/nemoretriever-parse/v1
```

---

## Local Development

```bash
# Install dependencies
pip install -r 1_session-install-dependencies/requirements.txt

# Set configuration via environment variables (or use the UI config tab)
export CAI_INFERENCE_TOKEN="your-token-here"
export CAI_OCR_ENDPOINT_URL="https://..."
export CAI_LLM_ENDPOINT_URL="https://..."
export CAI_LLM_MODEL_ID="meta/llama-3.3-70b-instruct"

# Run the app
streamlit run 2_application/app.py
```

---

## Application Tabs

### 🔍 Image Analysis
The main working tab. Select a use case, choose a processing mode, pick an image, and click **Process Image**. For the OCR → LLM Pipeline mode, both the raw OCR output and the final LLM analysis are displayed.

### 🖼️ Upload & Manage Images
Upload your own images (PNG, JPG, JPEG, GIF, WEBP). View and delete images from the library. Uploaded images become available in the Analysis tab's image selector.

### ⚙️ Configuration
Configure your Cloudera AI Inference token and endpoint URLs through the UI. Test connections directly from the form. Settings are saved to disk and persist across sessions.

### ℹ️ About
Overview of the application, architecture explanation, and model recommendations.

---

## Project Structure

```
.
├── 1_session-install-dependencies/
│   ├── requirements.txt          # Python dependencies
│   └── install.py                # CML job script: pip install
├── 2_application/
│   ├── app.py                    # Streamlit application (all UI + logic)
│   └── start-app.py              # CML application entry point
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
