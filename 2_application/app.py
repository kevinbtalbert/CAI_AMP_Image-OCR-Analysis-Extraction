import os
import json
import base64
import requests
import streamlit as st
from openai import OpenAI
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

APP_TITLE = "Image OCR & Analysis — Cloudera AI Inference"
DATA_DIR = Path(os.getenv("CDSW_HOME", "/home/cdsw")) / "data"
CONFIG_PATH = Path(os.getenv("CDSW_HOME", "/home/cdsw")) / ".cai_inference_config.json"
EXAMPLES_DIR = Path(__file__).parent.parent / "data" / "examples"

# The two models used for all use cases
LLM_MODEL_LABEL = "Llama 3.3 70B Instruct"
LLM_MODEL_ID = "meta/llama-3.3-70b-instruct"

OCR_MODEL_LABEL = "NeMo Retriever-Parse"
OCR_MODEL_ID = "nemoretriever-parse"

USE_CASES = [
    "Transcribing Typed Text",
    "Transcribing Handwritten Text",
    "Transcribing Forms",
    "Complicated Document QA",
    "Unstructured Information → JSON",
    "User Defined",
]

USE_CASE_PROMPTS = {
    "Transcribing Typed Text": "Transcribe all typed text from this document exactly as it appears.",
    "Transcribing Handwritten Text": "Transcribe all handwritten text from this document exactly as it appears.",
    "Transcribing Forms": "Transcribe this form exactly, preserving field labels and their values.",
    "Complicated Document QA": "Answer the following question based on the document:",
    "Unstructured Information → JSON": (
        "Convert the content of this document into well-structured JSON. "
        "Identify logical fields and group related information."
    ),
    "User Defined": "",
}

USE_CASE_DESCRIPTIONS = {
    "Transcribing Typed Text": "Extract printed or typed text from scanned documents, PDFs, or screenshots.",
    "Transcribing Handwritten Text": "Convert handwritten notes into searchable, editable text.",
    "Transcribing Forms": "Extract structured data from forms, preserving field labels and values.",
    "Complicated Document QA": "Ask a specific question and receive an answer grounded in the document.",
    "Unstructured Information → JSON": "Turn unstructured document content into machine-readable JSON.",
    "User Defined": "Write a fully custom prompt to instruct the model however you like.",
}

DEFAULT_EXAMPLE_IMAGES = {
    "Transcribing Typed Text": "ex1-stack_overflow.png",
    "Transcribing Handwritten Text": "ex2-school_notes.png",
    "Transcribing Forms": "ex3-vehicle_form.jpeg",
    "Complicated Document QA": "ex4-doc_qa.jpeg",
    "Unstructured Information → JSON": "ex5-org_chart.jpeg",
}

IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "gif", "webp")

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def default_config() -> dict:
    return {
        "inference_token": os.getenv("CAI_INFERENCE_TOKEN", ""),
        "ocr_endpoint_url": os.getenv("CAI_OCR_ENDPOINT_URL", ""),
        "llm_endpoint_url": os.getenv("CAI_LLM_ENDPOINT_URL", ""),
        "llm_model_id": LLM_MODEL_ID,
        "ocr_model": OCR_MODEL_ID,
        "processing_mode": "ocr_pipeline",  # "ocr_pipeline" | "vision_llm" | "ocr_only"
        "max_tokens": 4096,
    }


def load_config() -> dict:
    cfg = default_config()
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text())
            cfg.update(saved)
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def get_base64_encoded_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_media_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower().lstrip(".")
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")


def get_image_files(directory: str) -> list:
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(
        f.name for f in d.iterdir()
        if f.is_file() and f.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# NeMo Retriever-Parse OCR
# ---------------------------------------------------------------------------

def call_nemoretriever_parse(image_path: str, cfg: dict) -> str:
    """
    Calls the NVIDIA NeMo Retriever-Parse NIM endpoint.
    The NIM accepts a file via multipart form-data and returns
    structured text extraction results.
    """
    base_url = cfg["ocr_endpoint_url"].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    url = f"{base_url}/parse"
    token = cfg["inference_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, get_media_type(image_path))}
        resp = requests.post(url, headers=headers, files=files, timeout=120)

    resp.raise_for_status()
    result = resp.json()

    # Aggregate all extracted text blocks in document order
    blocks = result.get("data", [])
    if not blocks:
        # Some deployments wrap in a different key
        blocks = result.get("elements", result.get("content", []))

    text_parts = []
    for block in blocks:
        content = block.get("content", block.get("text", ""))
        if content and content.strip():
            text_parts.append(content.strip())

    if not text_parts:
        # Fallback: return raw JSON string if structure is unexpected
        return json.dumps(result, indent=2)

    return "\n\n".join(text_parts)


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

def call_paddleocr(image_path: str, cfg: dict) -> str:
    """
    Calls a PaddleOCR NIM endpoint.
    Accepts an image file via multipart form-data and returns
    detected text regions.
    """
    base_url = cfg["ocr_endpoint_url"].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    url = f"{base_url}/ocr"
    token = cfg["inference_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, get_media_type(image_path))}
        resp = requests.post(url, headers=headers, files=files, timeout=120)

    resp.raise_for_status()
    result = resp.json()

    # PaddleOCR returns a list of text regions with confidence scores
    regions = result.get("results", result.get("data", []))
    texts = []
    for region in regions:
        text = region.get("text", region.get("content", ""))
        if text and text.strip():
            texts.append(text.strip())

    if not texts:
        return json.dumps(result, indent=2)

    return "\n".join(texts)


# ---------------------------------------------------------------------------
# OCR dispatcher
# ---------------------------------------------------------------------------

def call_ocr(image_path: str, cfg: dict) -> str:
    model = cfg.get("ocr_model", "nemoretriever-parse")
    if model == "paddleocr":
        return call_paddleocr(image_path, cfg)
    return call_nemoretriever_parse(image_path, cfg)


# ---------------------------------------------------------------------------
# LLM (OpenAI-compatible)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, cfg: dict, system_prompt: str = None) -> str:
    """
    Calls a Cloudera AI Inference LLM endpoint using the OpenAI-compatible API.
    """
    client = OpenAI(
        base_url=cfg["llm_endpoint_url"],
        api_key=cfg["inference_token"],
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=cfg["llm_model_id"],
        messages=messages,
        max_tokens=cfg.get("max_tokens", 4096),
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Vision LLM (OpenAI-compatible multimodal)
# ---------------------------------------------------------------------------

def call_vision_llm(image_path: str, instruction: str, cfg: dict) -> str:
    """
    Calls a vision-capable LLM on Cloudera AI Inference.
    Uses the standard OpenAI multimodal message format (image_url with base64).
    """
    client = OpenAI(
        base_url=cfg["llm_endpoint_url"],
        api_key=cfg["inference_token"],
    )

    b64 = get_base64_encoded_image(image_path)
    media_type = get_media_type(image_path)
    data_url = f"data:{media_type};base64,{b64}"

    response = client.chat.completions.create(
        model=cfg["llm_model_id"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": instruction},
                ],
            }
        ],
        max_tokens=cfg.get("max_tokens", 4096),
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_image(
    image_path: str,
    instruction: str,
    cfg: dict,
) -> dict:
    """
    Routes the request through the configured processing pipeline.

    Returns a dict with keys:
        mode        – which pipeline ran
        ocr_text    – raw OCR output (None if not used)
        llm_output  – final LLM response (None if ocr_only)
        final       – the display-ready final answer
    """
    mode = cfg.get("processing_mode", "ocr_pipeline")

    if mode == "ocr_only":
        ocr_text = call_ocr(image_path, cfg)
        return {"mode": mode, "ocr_text": ocr_text, "llm_output": None, "final": ocr_text}

    if mode == "vision_llm":
        result = call_vision_llm(image_path, instruction, cfg)
        return {"mode": mode, "ocr_text": None, "llm_output": result, "final": result}

    # Default: ocr_pipeline — OCR first, then LLM
    ocr_text = call_ocr(image_path, cfg)
    system_prompt = (
        "You are a document analysis assistant. "
        "The user has provided text extracted from an image via OCR. "
        "Use that extracted text to fulfil the user's instruction accurately."
    )
    prompt = (
        f"The following text was extracted from a document image:\n\n"
        f"---\n{ocr_text}\n---\n\n"
        f"Instruction: {instruction}"
    )
    llm_output = call_llm(prompt, cfg, system_prompt=system_prompt)
    return {
        "mode": mode,
        "ocr_text": ocr_text,
        "llm_output": llm_output,
        "final": llm_output,
    }


# ---------------------------------------------------------------------------
# Connection tester
# ---------------------------------------------------------------------------

def test_llm_connection(cfg: dict) -> tuple[bool, str]:
    try:
        resp = call_llm("Reply with: OK", cfg)
        return True, f"Connected. Model responded: {resp[:80]}"
    except Exception as e:
        return False, str(e)


def test_ocr_connection(cfg: dict) -> tuple[bool, str]:
    base_url = cfg.get("ocr_endpoint_url", "").rstrip("/")
    if not base_url:
        return False, "OCR endpoint URL not configured."
    token = cfg.get("inference_token", "")
    if not token:
        return False, "Inference token not configured."
    # Ping the health/metrics endpoint
    check_url = base_url.rstrip("/v1").rstrip("/") + "/v1/metrics"
    try:
        r = requests.get(
            check_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if r.status_code < 400:
            return True, f"OCR endpoint reachable (HTTP {r.status_code})."
        return False, f"OCR endpoint returned HTTP {r.status_code}."
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("📄 Image OCR & Analysis with Cloudera AI Inference")

# Initialise session state
if "config" not in st.session_state:
    st.session_state.config = load_config()
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

cfg = st.session_state.config

# ─── Tab definitions ───────────────────────────────────────────────────────
tab_analysis, tab_images, tab_config, tab_about = st.tabs(
    ["🔍 Image Analysis", "🖼️ Upload & Manage Images", "⚙️ Configuration", "ℹ️ About"]
)


# ===========================================================================
# TAB 1 — IMAGE ANALYSIS
# ===========================================================================
with tab_analysis:
    # Warn if not configured
    token_ok = bool(cfg.get("inference_token"))
    llm_ok = bool(cfg.get("llm_endpoint_url"))
    ocr_ok = bool(cfg.get("ocr_endpoint_url")) or cfg.get("processing_mode") == "vision_llm"

    if not token_ok or not llm_ok:
        st.warning(
            "⚠️ Cloudera AI Inference is not yet configured. "
            "Please visit the **⚙️ Configuration** tab to enter your token and endpoint URLs."
        )

    col_ctrl, col_result = st.columns([1, 2])

    with col_ctrl:
        st.header("Analysis Settings")

        selected_use_case = st.selectbox("Use Case:", USE_CASES)
        st.caption(f"**{USE_CASE_DESCRIPTIONS[selected_use_case]}**")

        processing_mode = "ocr_pipeline"

        # Build the instruction
        if selected_use_case == "User Defined":
            instruction_text = st.text_area("Custom Prompt:", value="", height=120)
        else:
            instruction_text = USE_CASE_PROMPTS[selected_use_case]
            if selected_use_case == "Complicated Document QA":
                question = st.text_input("Your question about the document:")
                if question:
                    instruction_text = f"{instruction_text} {question}"

        # Image source
        if selected_use_case != "User Defined":
            img_src = st.radio("Image Source:", ["Use uploaded image", "Use example image"])
        else:
            img_src = "Use uploaded image"

        image_path = None
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if img_src == "Use uploaded image":
            image_files = get_image_files(str(DATA_DIR))
            if image_files:
                img_name = st.selectbox("Select an uploaded image:", image_files)
                image_path = str(DATA_DIR / img_name)
                st.image(image_path, width=260)
            else:
                st.info("No uploaded images yet — head to the **🖼️ Upload & Manage Images** tab.")
        else:
            example_file = DEFAULT_EXAMPLE_IMAGES.get(selected_use_case)
            if example_file:
                image_path = str(EXAMPLES_DIR / example_file)
                if Path(image_path).exists():
                    if f"enlarged_{example_file}" not in st.session_state:
                        st.session_state[f"enlarged_{example_file}"] = False
                    if st.button("Toggle Image Size", key=f"toggle_{example_file}"):
                        st.session_state[f"enlarged_{example_file}"] = (
                            not st.session_state[f"enlarged_{example_file}"]
                        )
                    w = 600 if st.session_state[f"enlarged_{example_file}"] else 220
                    st.image(image_path, width=w)
                else:
                    st.warning(f"Example image not found: {example_file}")
                    image_path = None

        ready = (
            image_path
            and instruction_text
            and token_ok
            and llm_ok
            and (ocr_ok or processing_mode == "vision_llm")
        )

        process_btn = st.button(
            "🚀 Process Image",
            disabled=not ready,
            use_container_width=True,
            type="primary",
        )

    with col_result:
        st.header("Results")

        if process_btn and image_path and instruction_text:
            # Override mode in a local copy so the UI setting takes effect
            run_cfg = {**cfg, "processing_mode": processing_mode}

            with st.spinner("Processing…"):
                try:
                    result = process_image(image_path, instruction_text, run_cfg)
                except Exception as e:
                    st.error(f"**Error:** {e}")
                    result = None

            if result:
                if result["mode"] == "ocr_pipeline" and result["ocr_text"]:
                    with st.expander("📝 OCR Extracted Text (Stage 1)", expanded=False):
                        st.code(result["ocr_text"], language="text")

                st.subheader(
                    "🤖 LLM Analysis" if result["mode"] != "ocr_only" else "📄 Extracted Text"
                )
                st.code(result["final"], language="markdown")

        elif not ready and process_btn:
            st.warning("Please configure your endpoints in the **⚙️ Configuration** tab first.")
        else:
            st.info("Configure the options on the left and click **🚀 Process Image** to begin.")


# ===========================================================================
# TAB 2 — UPLOAD & MANAGE IMAGES
# ===========================================================================
with tab_images:
    st.header("Upload & Manage Images")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    uploaded_file = st.file_uploader(
        "Upload an image (PNG, JPG, JPEG, GIF, WEBP):",
        type=list(IMAGE_EXTENSIONS),
    )

    if uploaded_file and uploaded_file.name != st.session_state.uploaded_file_name:
        save_path = DATA_DIR / uploaded_file.name
        save_path.write_bytes(uploaded_file.getbuffer())
        st.success(f"Saved: {uploaded_file.name}")
        st.session_state.uploaded_file_name = uploaded_file.name
        st.rerun()

    if not uploaded_file:
        st.session_state.uploaded_file_name = None

    col_list, col_preview = st.columns([1, 2])

    with col_list:
        st.subheader("Stored Images")
        image_files = get_image_files(str(DATA_DIR))
        if image_files:
            for fname in image_files:
                fpath = str(DATA_DIR / fname)
                c_name, c_view, c_del = st.columns([3, 1, 1])
                with c_name:
                    st.text(fname)
                with c_view:
                    if st.button("View", key=f"view_{fname}"):
                        st.session_state.selected_image = fpath
                with c_del:
                    if st.button("Del", key=f"del_{fname}"):
                        Path(fpath).unlink(missing_ok=True)
                        if st.session_state.selected_image == fpath:
                            st.session_state.selected_image = None
                        st.rerun()
        else:
            st.info("No images uploaded yet.")

    with col_preview:
        if st.session_state.selected_image and Path(st.session_state.selected_image).exists():
            st.subheader("Preview")
            st.image(st.session_state.selected_image, use_column_width=True)


# ===========================================================================
# TAB 3 — CONFIGURATION
# ===========================================================================
with tab_config:
    st.header("Cloudera AI Inference — Configuration")

    st.markdown(
        """
        Enter your **Cloudera AI Inference Service** endpoint URLs and token below.  
        Settings are saved to disk and persist across sessions.

        **Endpoint URL format:**
        ```
        https://ml-<workspace-id>.<env>.<tenant>.cloudera.site/namespaces/serving-default/endpoints/<endpoint-name>/v1
        ```
        """
    )

    st.divider()

    # ── Models in use ───────────────────────────────────────────────────────
    st.info(
        f"**OCR Model:** {OCR_MODEL_LABEL}  ·  **Analysis Model:** {LLM_MODEL_LABEL}\n\n"
        "These are the recommended models from the Cloudera AI Inference catalog "
        "for document OCR and intelligent analysis respectively."
    )

    st.divider()

    # ── Authentication ──────────────────────────────────────────────────────
    st.subheader("🔑 Token")
    new_token = st.text_input(
        "CDP / AI Inference Token:",
        value=cfg.get("inference_token", ""),
        type="password",
        help=(
            "Your Cloudera Data Platform JWT token. "
            "The same token is used for both the OCR and LLM endpoints. "
            "You can also set `CAI_INFERENCE_TOKEN` as an environment variable."
        ),
    )

    st.divider()

    # ── NeMo Retriever-Parse endpoint ───────────────────────────────────────
    st.subheader(f"📷 {OCR_MODEL_LABEL} Endpoint")

    new_ocr_url = st.text_input(
        "OCR Endpoint URL:",
        value=cfg.get("ocr_endpoint_url", ""),
        placeholder="https://ml-xxxxx.cloudera.site/namespaces/serving-default/endpoints/nemoretriever-parse/v1",
        help="You can also set `CAI_OCR_ENDPOINT_URL` as an environment variable.",
    )

    col_test_ocr, col_ocr_status = st.columns([1, 3])
    with col_test_ocr:
        test_ocr_btn = st.button("Test OCR Connection", use_container_width=True)
    with col_ocr_status:
        if test_ocr_btn:
            test_cfg = {**cfg, "ocr_endpoint_url": new_ocr_url, "inference_token": new_token}
            ok, msg = test_ocr_connection(test_cfg)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.divider()

    # ── Llama 3.3 70B endpoint ──────────────────────────────────────────────
    st.subheader(f"🤖 {LLM_MODEL_LABEL} Endpoint")

    new_llm_url = st.text_input(
        "LLM Endpoint URL:",
        value=cfg.get("llm_endpoint_url", ""),
        placeholder="https://ml-xxxxx.cloudera.site/namespaces/serving-default/endpoints/llama-3-3-70b/v1",
        help="You can also set `CAI_LLM_ENDPOINT_URL` as an environment variable.",
    )

    col_test_llm, col_llm_status = st.columns([1, 3])
    with col_test_llm:
        test_llm_btn = st.button("Test LLM Connection", use_container_width=True)
    with col_llm_status:
        if test_llm_btn:
            test_cfg = {
                **cfg,
                "llm_endpoint_url": new_llm_url,
                "llm_model_id": LLM_MODEL_ID,
                "inference_token": new_token,
            }
            ok, msg = test_llm_connection(test_cfg)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.divider()

    # ── Save ────────────────────────────────────────────────────────────────
    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        updated = {
            **cfg,
            "inference_token": new_token,
            "ocr_endpoint_url": new_ocr_url,
            "ocr_model": OCR_MODEL_ID,
            "llm_endpoint_url": new_llm_url,
            "llm_model_id": LLM_MODEL_ID,
            "processing_mode": cfg.get("processing_mode", "ocr_pipeline"),
            "max_tokens": cfg.get("max_tokens", 4096),
        }
        save_config(updated)
        st.session_state.config = updated
        cfg = updated
        st.success("✅ Configuration saved successfully!")
        st.rerun()

    st.divider()
    st.subheader("📋 Environment Variable Reference")
    st.markdown(
        """
        You can pre-configure the app via CML environment variables instead of this form:

        | Variable | Description |
        |---|---|
        | `CAI_INFERENCE_TOKEN` | CDP JWT token (used for all endpoints) |
        | `CAI_OCR_ENDPOINT_URL` | Full URL of the NeMo Retriever-Parse endpoint |
        | `CAI_LLM_ENDPOINT_URL` | Full URL of the Llama 3.3 70B endpoint |
        """
    )


# ===========================================================================
# TAB 4 — ABOUT
# ===========================================================================
with tab_about:
    st.header("About This Application")

    col_text, col_img = st.columns([3, 1])

    with col_text:
        st.subheader("What This App Does")
        st.write(
            """
            This application lets you analyse images and documents — transcribing typed or
            handwritten text, extracting form data, answering document questions, and
            converting unstructured content to JSON — all powered by models running on
            **Cloudera's AI Inference Service**.

            Unlike traditional OCR tools, the pipeline combines a dedicated document
            extraction model with a large language model, giving you both accurate text
            extraction and intelligent, context-aware analysis in a single step.
            """
        )

        st.subheader("Architecture: OCR → LLM Pipeline")
        st.write(
            """
            The default processing mode uses a **two-stage pipeline**:

            1. **Stage 1 — OCR / Text Extraction**  
               An image is sent to the **NeMo Retriever-Parse** NIM. This NVIDIA model is
               specifically designed for documents and images: it extracts formatted text,
               identifies semantic block types (headings, paragraphs, tables, captions), and
               preserves document structure.

            2. **Stage 2 — LLM Analysis**  
               The extracted text is passed to a large language model
               (e.g. **Llama 3.3 70B Instruct**) together with your instruction. The LLM
               then performs the requested task — transcription clean-up, question answering,
               JSON structuring, or any custom prompt.

            Alternatively, if you have a **vision-capable LLM** deployed (e.g. a Llama 3.2
            Vision variant), you can switch to **Vision LLM** mode to send the image
            directly and skip the OCR stage.
            """
        )

        st.subheader("Models Used")
        st.markdown(
            """
            | Stage | Model | Why |
            |---|---|---|
            | **OCR / Text Extraction** | **NeMo Retriever-Parse** | Purpose-built for document images; extracts formatted text with semantic labels (headings, paragraphs, tables) |
            | **Analysis / Reasoning** | **Llama 3.3 70B Instruct** | The strongest general-purpose model in the Cloudera AI Inference catalog; handles all six use cases with high accuracy |

            Both models run entirely within your Cloudera environment — no external API calls.
            """
        )

        st.subheader("Supported Use Cases")
        for uc, desc in USE_CASE_DESCRIPTIONS.items():
            st.markdown(f"- **{uc}** — {desc}")

    with col_img:
        st.image(
            "https://www.cloudera.com/content/dam/www/marketing/images/logos/cloudera-logo-tm.png",
            use_column_width=True,
            caption="Powered by Cloudera AI Inference",
        )
        st.markdown("---")
        st.markdown(
            """
            **Key Technologies**
            - [Streamlit](https://streamlit.io/)
            - [NVIDIA NIM](https://developer.nvidia.com/nim)
            - [OpenAI Python SDK](https://github.com/openai/openai-python)
            - [Cloudera AI](https://www.cloudera.com/products/machine-learning.html)
            """
        )
