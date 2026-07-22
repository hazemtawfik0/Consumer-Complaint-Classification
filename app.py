from __future__ import annotations

import csv
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MAX_LENGTH = 192


def _contains_model_files(directory: Path) -> bool:
    """Return True when a folder looks like a saved Hugging Face model."""
    if not directory.is_dir():
        return False

    has_config = (directory / "config.json").is_file()
    has_tokenizer = any(
        (directory / filename).is_file()
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        )
    )
    has_weights = any(
        (directory / filename).is_file()
        for filename in (
            "model.safetensors",
            "pytorch_model.bin",
        )
    )
    return has_config and has_tokenizer and has_weights


def find_model_directory() -> Path:
    """
    Locate the final model after the Kaggle ZIP is extracted.

    Supported layouts:
      project/fine_tuned_transformer/
      project/consumer_complaint_transformer_final_60k/fine_tuned_transformer/
      A path supplied through the MODEL_DIR environment variable.
    """
    environment_path = os.getenv("MODEL_DIR")
    preferred_candidates: list[Path] = []

    if environment_path:
        preferred_candidates.append(Path(environment_path).expanduser())

    preferred_candidates.extend(
        [
            BASE_DIR / "fine_tuned_transformer",
            BASE_DIR
            / "consumer_complaint_transformer_final_60k"
            / "fine_tuned_transformer",
            BASE_DIR / "model" / "fine_tuned_transformer",
            BASE_DIR / "model",
        ]
    )

    for candidate in preferred_candidates:
        candidate = candidate.resolve()
        if _contains_model_files(candidate):
            return candidate

    # Fall back to a recursive search, preferring the explicitly named final folder.
    discovered = [
        path
        for path in BASE_DIR.rglob("*")
        if path.is_dir() and _contains_model_files(path)
    ]

    named_final = [
        path
        for path in discovered
        if path.name.lower() == "fine_tuned_transformer"
    ]

    if named_final:
        return sorted(named_final, key=lambda path: len(path.parts))[0]

    if len(discovered) == 1:
        return discovered[0]

    if len(discovered) > 1:
        choices = "\n".join(f"  - {path}" for path in discovered)
        raise RuntimeError(
            "Several model folders were found. Set MODEL_DIR to the final "
            f"fine_tuned_transformer folder.\n{choices}"
        )

    raise FileNotFoundError(
        "The trained model was not found.\n\n"
        "Extract consumer_complaint_transformer_final_60k.zip into this "
        "project folder. The expected folder is:\n"
        f"{BASE_DIR / 'fine_tuned_transformer'}\n\n"
        "That folder must contain config.json, tokenizer files, and "
        "model.safetensors."
    )


def find_experiment_file(filename: str, model_directory: Path) -> Path | None:
    candidates = [
        model_directory.parent / filename,
        BASE_DIR / filename,
        BASE_DIR
        / "consumer_complaint_transformer_final_60k"
        / filename,
    ]
    return next((path for path in candidates if path.is_file()), None)


def load_experiment_configuration(model_directory: Path) -> dict[str, Any]:
    config_path = find_experiment_file(
        "experiment_config.json",
        model_directory,
    )
    if config_path is None:
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def load_metrics(model_directory: Path) -> dict[str, float]:
    metrics_path = find_experiment_file(
        "final_transformer_metrics.csv",
        model_directory,
    )
    if metrics_path is None:
        return {}

    try:
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as file:
            row = next(csv.DictReader(file), None)

        if not row:
            return {}

        metrics: dict[str, float] = {}
        for key, value in row.items():
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        return metrics
    except OSError:
        return {}


def friendly_label(label: str) -> str:
    """Convert labels such as credit_reporting into Credit Reporting."""
    return str(label).replace("_", " ").strip().title()


def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\b[xX]{2,}\b", " redacted ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


MODEL_DIR = find_model_directory()
EXPERIMENT_CONFIG = load_experiment_configuration(MODEL_DIR)
METRICS = load_metrics(MODEL_DIR)

MAX_LENGTH = int(
    EXPERIMENT_CONFIG.get(
        "max_length",
        EXPERIMENT_CONFIG.get(
            "transformer_max_length",
            DEFAULT_MAX_LENGTH,
        ),
    )
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Loading model from: {MODEL_DIR}")
print(f"Device: {DEVICE}")

TOKENIZER = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)

MODEL = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)

MODEL.to(DEVICE)
MODEL.eval()

RAW_ID_TO_LABEL = MODEL.config.id2label or {}
ID_TO_LABEL = {
    int(index): str(label)
    for index, label in RAW_ID_TO_LABEL.items()
}

if not ID_TO_LABEL:
    ID_TO_LABEL = {
        index: f"Category {index}"
        for index in range(MODEL.config.num_labels)
    }

DISPLAY_LABELS = {
    index: friendly_label(label)
    for index, label in ID_TO_LABEL.items()
}


def confidence_level(confidence: float) -> tuple[str, str]:
    if confidence >= 0.75:
        return "High confidence", "confidence-high"
    if confidence >= 0.55:
        return "Medium confidence", "confidence-medium"
    return "Low confidence", "confidence-low"


def empty_prediction_card() -> str:
    return """
    <div class="empty-card">
        <div class="empty-icon">⌁</div>
        <div class="empty-title">Prediction will appear here</div>
        <div class="empty-copy">
            Enter a consumer complaint and select <b>Classify complaint</b>.
        </div>
    </div>
    """


def predict_complaint(
    complaint: str,
) -> tuple[str, dict[str, float], dict[str, Any]]:
    cleaned = normalize_text(complaint)

    if len(cleaned) < 10:
        raise gr.Error(
            "Enter a meaningful complaint containing at least 10 characters."
        )

    start_time = time.perf_counter()

    encoded = TOKENIZER(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        return_token_type_ids=False,
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    encoded.pop("token_type_ids", None)

    with torch.inference_mode():
        logits = MODEL(**encoded).logits
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    best_index = int(torch.argmax(probabilities).item())
    best_probability = float(probabilities[best_index].item())
    raw_label = ID_TO_LABEL[best_index]
    display_label = DISPLAY_LABELS[best_index]
    level_text, level_class = confidence_level(best_probability)

    sorted_indices = torch.argsort(
        probabilities,
        descending=True,
    ).tolist()

    probability_dictionary = {
        DISPLAY_LABELS[index]: float(probabilities[index].item())
        for index in sorted_indices
    }

    top_three = [
        {
            "category": DISPLAY_LABELS[index],
            "score": round(float(probabilities[index].item()), 6),
        }
        for index in sorted_indices[:3]
    ]

    result_html = f"""
    <div class="result-card">
        <div class="result-kicker">Predicted complaint category</div>
        <div class="result-label">{html.escape(display_label)}</div>
        <div class="result-meta-row">
            <span class="confidence-pill {level_class}">
                {level_text}
            </span>
            <span class="confidence-number">
                {best_probability:.1%}
            </span>
        </div>
        <div class="result-note">
            Internal category code:
            <code>{html.escape(raw_label)}</code>
        </div>
    </div>
    """

    details = {
        "predicted_category": raw_label,
        "display_category": display_label,
        "confidence": round(best_probability, 6),
        "top_3": top_three,
        "input_characters": len(cleaned),
        "input_tokens_after_truncation": int(encoded["input_ids"].shape[1]),
        "maximum_tokens": MAX_LENGTH,
        "device": str(DEVICE),
        "inference_time_ms": round(elapsed_ms, 2),
        "confidence_note": (
            "The score is a softmax model output and is not a guarantee."
        ),
    }

    return result_html, probability_dictionary, details


def reset_interface() -> tuple[str, str, dict[str, float], dict[str, Any]]:
    return "", empty_prediction_card(), {}, {}


accuracy = METRICS.get("accuracy")
macro_f1 = METRICS.get("f1_macro")
trained_rows = EXPERIMENT_CONFIG.get(
    "modeling_rows",
    "60,000",
)

accuracy_text = (
    f"{accuracy:.1%}" if isinstance(accuracy, float) else "Final model"
)
macro_f1_text = (
    f"{macro_f1:.1%}" if isinstance(macro_f1, float) else "DistilBERT"
)
device_text = "NVIDIA GPU" if DEVICE.type == "cuda" else "CPU"

CSS = """
:root {
    --brand-blue: #2563eb;
    --brand-cyan: #06b6d4;
    --ink: #0f172a;
    --muted: #64748b;
    --panel: rgba(255, 255, 255, 0.86);
    --line: rgba(148, 163, 184, 0.28);
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 24px 18px 48px !important;
    background:
        radial-gradient(circle at 5% 0%, rgba(37,99,235,.13), transparent 34%),
        radial-gradient(circle at 95% 10%, rgba(6,182,212,.12), transparent 30%);
}

.hero {
    border: 1px solid var(--line);
    border-radius: 28px;
    padding: 34px;
    margin-bottom: 20px;
    background:
        linear-gradient(135deg, rgba(15,23,42,.98), rgba(30,64,175,.94));
    box-shadow: 0 24px 70px rgba(15, 23, 42, .18);
    color: white;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.18);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: .03em;
}

.hero h1 {
    margin: 18px 0 10px;
    font-size: clamp(32px, 5vw, 54px);
    line-height: 1.02;
    letter-spacing: -.04em;
    color: white !important;
}

.hero p {
    margin: 0;
    max-width: 760px;
    color: rgba(255,255,255,.79);
    font-size: 17px;
    line-height: 1.7;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 26px;
}

.stat-card {
    border-radius: 18px;
    padding: 15px 17px;
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.14);
}

.stat-value {
    font-size: 23px;
    font-weight: 800;
    color: white;
}

.stat-label {
    margin-top: 4px;
    font-size: 12px;
    color: rgba(255,255,255,.66);
    text-transform: uppercase;
    letter-spacing: .08em;
}

.panel {
    border: 1px solid var(--line) !important;
    border-radius: 24px !important;
    padding: 18px !important;
    background: var(--panel) !important;
    box-shadow: 0 15px 45px rgba(15, 23, 42, .08) !important;
}

.section-label {
    font-weight: 800;
    font-size: 14px;
    color: var(--ink);
    margin-bottom: 8px;
}

.primary-button {
    background: linear-gradient(135deg, var(--brand-blue), var(--brand-cyan)) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 12px 25px rgba(37,99,235,.24) !important;
}

.primary-button:hover {
    transform: translateY(-1px);
    filter: brightness(1.03);
}

.result-card,
.empty-card {
    min-height: 205px;
    border-radius: 22px;
    padding: 26px;
    border: 1px solid var(--line);
}

.result-card {
    background:
        linear-gradient(145deg, rgba(37,99,235,.10), rgba(6,182,212,.06)),
        white;
}

.empty-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    background: rgba(248,250,252,.78);
    color: var(--muted);
}

.empty-icon {
    width: 48px;
    height: 48px;
    display: grid;
    place-items: center;
    border-radius: 15px;
    background: white;
    color: var(--brand-blue);
    font-size: 27px;
    box-shadow: 0 10px 28px rgba(15,23,42,.09);
}

.empty-title {
    color: var(--ink);
    font-weight: 800;
    margin-top: 14px;
}

.empty-copy {
    margin-top: 5px;
    max-width: 360px;
}

.result-kicker {
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .09em;
    font-weight: 800;
}

.result-label {
    margin-top: 12px;
    color: var(--ink);
    font-size: clamp(28px, 4vw, 42px);
    font-weight: 900;
    letter-spacing: -.035em;
}

.result-meta-row {
    display: flex;
    align-items: center;
    gap: 11px;
    margin-top: 18px;
}

.confidence-pill {
    display: inline-flex;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 800;
}

.confidence-high {
    background: #dcfce7;
    color: #166534;
}

.confidence-medium {
    background: #fef3c7;
    color: #92400e;
}

.confidence-low {
    background: #fee2e2;
    color: #991b1b;
}

.confidence-number {
    font-size: 22px;
    font-weight: 900;
    color: var(--ink);
}

.result-note {
    margin-top: 20px;
    color: var(--muted);
    font-size: 13px;
}

.footer-note {
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    padding: 16px 0 4px;
}

@media (max-width: 720px) {
    .hero {
        padding: 24px;
        border-radius: 22px;
    }

    .stats-grid {
        grid-template-columns: 1fr;
    }
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=CSS,
    title="Consumer Complaint Classifier",
) as demo:
    gr.HTML(
        f"""
        <section class="hero">
            <div class="hero-badge">
                ✦ NLP · DISTILBERT · LOCAL INFERENCE
            </div>
            <h1>Consumer Complaint Classifier</h1>
            <p>
                Classify financial consumer complaints into five operational
                categories using the final Transformer trained on a stratified
                dataset of {html.escape(str(trained_rows))} records.
            </p>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{accuracy_text}</div>
                    <div class="stat-label">Test accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{macro_f1_text}</div>
                    <div class="stat-label">Macro F1</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{device_text}</div>
                    <div class="stat-label">Current device</div>
                </div>
            </div>
        </section>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=6, elem_classes=["panel"]):
            gr.HTML(
                '<div class="section-label">Complaint narrative</div>'
            )

            complaint_input = gr.Textbox(
                label="",
                placeholder=(
                    "Example: I found accounts on my credit report "
                    "that do not belong to me..."
                ),
                lines=11,
                max_lines=16,
                autofocus=True,
                elem_id="complaint-input",
            )

            with gr.Row():
                classify_button = gr.Button(
                    "Classify complaint",
                    variant="primary",
                    elem_classes=["primary-button"],
                )
                clear_button = gr.Button(
                    "Clear",
                    variant="secondary",
                )

            gr.Examples(
                examples=[
                    [
                        "I found accounts on my credit report that "
                        "do not belong to me."
                    ],
                    [
                        "The mortgage company charged a late fee "
                        "although I made the payment on time."
                    ],
                    [
                        "A debt collector keeps calling me about a "
                        "debt that I already paid."
                    ],
                    [
                        "My credit card company added a transaction "
                        "that I never authorized."
                    ],
                    [
                        "The bank closed my checking account and has "
                        "not returned the remaining balance."
                    ],
                ],
                inputs=complaint_input,
                label="Try an example",
            )

        with gr.Column(scale=5, elem_classes=["panel"]):
            gr.HTML(
                '<div class="section-label">Classification result</div>'
            )

            result_card = gr.HTML(
                value=empty_prediction_card()
            )

            probability_output = gr.Label(
                label="Category probabilities",
                num_top_classes=5,
            )

    with gr.Accordion(
        "Technical details",
        open=False,
    ):
        details_output = gr.JSON(
            label="Inference details"
        )

    gr.HTML(
        """
        <div class="footer-note">
            The displayed confidence is a softmax model score and should not
            be treated as a guarantee or a substitute for human review.
        </div>
        """
    )

    classify_event = classify_button.click(
        fn=predict_complaint,
        inputs=complaint_input,
        outputs=[
            result_card,
            probability_output,
            details_output,
        ],
        show_progress="minimal",
    )

    complaint_input.submit(
        fn=predict_complaint,
        inputs=complaint_input,
        outputs=[
            result_card,
            probability_output,
            details_output,
        ],
        show_progress="minimal",
    )

    clear_button.click(
        fn=reset_interface,
        inputs=None,
        outputs=[
            complaint_input,
            result_card,
            probability_output,
            details_output,
        ],
        queue=False,
    )


if __name__ == "__main__":
    server_name = os.getenv(
        "GRADIO_SERVER_NAME",
        "127.0.0.1",
    )
    server_port = int(
        os.getenv(
            "GRADIO_SERVER_PORT",
            "7860",
        )
    )

    demo.queue().launch(
        server_name=server_name,
        server_port=server_port,
        inbrowser=True,
        share=False,
        show_error=True,
    )
