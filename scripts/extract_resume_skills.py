"""
Resume Skill Extractor using yashpwr/resume-ner-bert-v2
=======================================================
Extracts skill entities from resume_text in a CSV file using a
BERT-based NER model fine-tuned on resume data.

Uses direct model inference with manual B-/I- tag merging (as
recommended by the model author) instead of the HuggingFace pipeline,
which fragments sub-tokens incorrectly for this model.

Requirements:
    pip install transformers torch pandas tqdm

Usage:
    python scripts/extract_resume_skills.py
    python scripts/extract_resume_skills.py --input data.csv --output results.csv
"""

import argparse
import ast
import re
import warnings
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_NAME       = "yashpwr/resume-ner-bert-v2"
INPUT_FILE       = "dataset/huggingface/train.csv"
OUTPUT_FILE      = "dataset/huggingface/train_with_skills.csv"
RESUME_COL       = "resume_text"
OUTPUT_COL       = "resume_skills"
MAX_LENGTH       = 512          # BERT's max token limit
CHUNK_OVERLAP    = 50           # overlapping tokens between chunks
SKILL_LABELS     = {"Skills"}   # entity label(s) to extract
BATCH_SIZE       = 8


# ─── MODEL LOADING ───────────────────────────────────────────────────────────

def load_model():
    """Load tokenizer + model for direct inference."""
    print(f"[1/4] Loading model: {MODEL_NAME} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Running on {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    labels = list(model.config.id2label.values())
    print(f"      Model labels: {labels}")
    return model, tokenizer, device


# ─── CORE EXTRACTION (direct inference, no pipeline) ─────────────────────────

def extract_entities_from_chunk(text: str, model, tokenizer, device) -> list:
    """
    Run NER on a single chunk using direct model inference.
    Uses manual B-/I- tag merging as recommended by the model author.

    Returns list of dicts: [{'text': ..., 'label': ...}, ...]
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Manual B-/I- entity merging (from HuggingFace model card)
    entities = []
    current_entity = None

    input_ids = inputs["input_ids"][0]
    for i, pred in enumerate(predictions[0]):
        label = model.config.id2label[pred.item()]
        token = tokenizer.convert_ids_to_tokens(input_ids[i].item())

        # Skip special tokens
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        if label.startswith("B-"):
            # Start of a new entity — save previous if any
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "tokens": [token],
                "label": label[2:],  # Remove 'B-' prefix
            }
        elif label.startswith("I-") and current_entity:
            # Continuation of current entity
            current_entity["tokens"].append(token)
        else:
            # O label — close current entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    # Don't forget the last entity
    if current_entity:
        entities.append(current_entity)

    # Reconstruct text from sub-tokens
    for ent in entities:
        ent["text"] = _merge_subtokens(ent["tokens"])

    return entities


def _merge_subtokens(tokens: list) -> str:
    """Merge BERT sub-tokens (## prefixed) back into readable text."""
    if not tokens:
        return ""

    result = []
    for token in tokens:
        if token.startswith("##"):
            # Sub-token: append without space
            result.append(token[2:])
        else:
            # New word: add space separator
            if result:
                result.append(" ")
            result.append(token)

    return "".join(result).strip()


# ─── CHUNKING FOR LONG TEXTS ────────────────────────────────────────────────

def chunk_text(text: str, tokenizer, max_length: int = MAX_LENGTH - 2,
               overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split long text into overlapping token-based chunks that fit
    within BERT's 512-token window (minus 2 for [CLS]/[SEP]).

    Returns a list of text strings.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_length:
        return [text]

    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_length, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_str = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_str)
        if end == len(token_ids):
            break
        start += max_length - overlap
    return chunks


# ─── SKILL EXTRACTION ────────────────────────────────────────────────────────

def clean_skill(skill: str) -> str:
    """Normalise a raw skill string."""
    # Remove stray punctuation but keep +, #, ., -, /
    skill = re.sub(r"[^\w\s\+\#\./\-]", " ", skill)
    skill = " ".join(skill.split())
    return skill.strip()


# Patterns that indicate a section header, not a skill
_HEADER_PATTERN = re.compile(
    r"^(skills?|technical skills?|core competenc|key skills?|"
    r"areas? of expertise|proficienc|qualifications?|"
    r"tools?|technologies?|summary|experience|education|"
    r"certifications?|projects?|interests?|hobbies?|"
    r"languages?|references?|objective|profile)\s*:?\s*$",
    re.IGNORECASE,
)


def _split_compound_entity(raw: str) -> list:
    """
    Split a compound NER entity into individual skills.

    The model often merges entire skill sections into one entity, e.g.:
      "Skills: Python, TensorFlow, SQL, Machine Learning"
    This function splits on commas, semicolons, pipes, bullet chars,
    and strips section headers like "Skills:" from the front.
    """
    # Strip leading section headers like "Skills:", "Technical Skills :"
    raw = re.sub(
        r"^(skills?|technical skills?|core competenc\w*|key skills?|"
        r"areas? of expertise|proficienc\w*|qualifications?|"
        r"tools? (?:and|&) technolog\w*|tools?|technologies?)\s*:?\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()

    if not raw:
        return []

    # Split on comma, semicolon, pipe, bullet, newline, or " and " / " & "
    parts = re.split(r"[,;|\u2022\n]+|\s+(?:and|&)\s+", raw)

    results = []
    for part in parts:
        part = clean_skill(part)
        if len(part) < 3:
            continue
        # Must contain at least one letter
        if not re.search(r"[a-zA-Z]", part):
            continue
        # Skip if it's just a section header
        if _HEADER_PATTERN.match(part):
            continue
        # Skip if it's too long to be a single skill (>60 chars = likely a sentence)
        if len(part) > 60:
            continue
        # Skip fragments: must have at least 2 alpha chars
        alpha_chars = sum(1 for c in part if c.isalpha())
        if alpha_chars < 3:
            continue
        # Skip obvious sub-token fragments (e.g. "ed and", "ing", "tion")
        if re.match(r"^(ed|ing|tion|ment|ness|ful|ble|ly|er|est|ive|ous|al)\b", part.lower()):
            continue
        results.append(part)

    return results


def extract_skills_from_text(text: str, model, tokenizer, device) -> list:
    """
    Extract skill entities from a (possibly long) resume text.
    Returns a deduplicated list of cleaned skill strings.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    chunks = chunk_text(text, tokenizer)
    seen = set()
    skills = []

    for chunk in chunks:
        try:
            entities = extract_entities_from_chunk(chunk, model, tokenizer, device)
        except Exception as e:
            print(f"      [WARN] Inference error on chunk: {e}")
            continue

        for ent in entities:
            if ent["label"] not in SKILL_LABELS:
                continue

            raw = ent["text"]

            # Post-process: split compound entities on delimiters
            individual_skills = _split_compound_entity(raw)

            for skill in individual_skills:
                key = skill.lower()
                if key not in seen:
                    seen.add(key)
                    skills.append(skill)

    return skills


# ─── DATAFRAME PROCESSING ───────────────────────────────────────────────────

def process_dataframe(df: pd.DataFrame, model, tokenizer, device) -> pd.DataFrame:
    """Process all rows and add the OUTPUT_COL column."""
    print(f"[3/4] Extracting skills from {len(df):,} records ...")

    all_skills = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing resumes"):
        skills = extract_skills_from_text(row[RESUME_COL], model, tokenizer, device)
        all_skills.append(skills)

    df = df.copy()
    df[OUTPUT_COL] = all_skills
    return df


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main(input_file: str, output_file: str):
    # 1. Load model
    model, tokenizer, device = load_model()

    # 2. Load data
    print(f"[2/4] Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"      Shape: {df.shape}")
    print(f"      Columns: {df.columns.tolist()}")

    if RESUME_COL not in df.columns:
        raise ValueError(
            f"Column '{RESUME_COL}' not found. Available: {df.columns.tolist()}"
        )

    # 3. Process
    df_out = process_dataframe(df, model, tokenizer, device)

    # 4. Save
    print(f"[4/4] Saving results to: {output_file}")
    df_out.to_csv(output_file, index=False)

    # ── Summary stats ───────────────────────────────────────────────────────
    skill_counts = df_out[OUTPUT_COL].apply(len)
    total_skills = skill_counts.sum()
    avg_skills = skill_counts.mean()
    empty_records = (skill_counts == 0).sum()
    all_unique = set(s.lower() for lst in df_out[OUTPUT_COL] for s in lst)

    print("\n" + "=" * 55)
    print("  EXTRACTION SUMMARY")
    print("=" * 55)
    print(f"  Total records processed : {len(df_out):,}")
    print(f"  Records with 0 skills   : {empty_records:,} ({empty_records / len(df_out) * 100:.1f}%)")
    print(f"  Total skill mentions    : {total_skills:,}")
    print(f"  Unique skills (global)  : {len(all_unique):,}")
    print(f"  Avg skills per resume   : {avg_skills:.2f}")
    print(f"  Min / Max per resume    : {skill_counts.min()} / {skill_counts.max()}")
    print("=" * 55)

    # Show a quick sample
    sample = df_out[[RESUME_COL, OUTPUT_COL]].head(3)
    for i, row in sample.iterrows():
        preview = str(row[RESUME_COL])[:120].replace("\n", " ")
        print(f"\n  [Row {i}] Resume preview: {preview}...")
        print(f"          Skills found  : {row[OUTPUT_COL]}")

    print(f"\nDone! Output saved to → {output_file}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract resume skills using BERT NER"
    )
    parser.add_argument("--input", default=INPUT_FILE, help="Input CSV file path")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output CSV file path")
    args = parser.parse_args()

    main(args.input, args.output)
