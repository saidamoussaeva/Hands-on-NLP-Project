"""
Preprocessing OHSUMED pour Information Retrieval
Simplifié : pas de stopwords removal (nuisible pour embeddings)
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def project_root() -> Path:
    # Chercher le répertoire contenant 'data/'
    current = Path(__file__).resolve()
    for parent in [current.parent, current.parents[1], current.parents[2]]:
        if (parent / "data").exists():
            return parent
    raise FileNotFoundError("Project root avec 'data/' introuvable")


MEDICAL_ABBREVIATIONS = {
    "mi": "myocardial infarction",
    "htn": "hypertension",
    "cad": "coronary artery disease",
    "cvd": "cardiovascular disease",
    "chf": "congestive heart failure",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "lvef": "left ventricular ejection fraction",
    "ace": "angiotensin converting enzyme",
    "arb": "angiotensin receptor blocker",
}


class IRPreprocessor:
    """Preprocessing minimal pour IR : normalisation + expansion"""

    def __init__(self, expand_abbreviations: bool = True):
        self.expand_abbreviations = expand_abbreviations

    def expand_medical_abbreviations(self, text: str) -> str:
        if not self.expand_abbreviations:
            return text
        t = text.lower()
        for abbrev, expansion in MEDICAL_ABBREVIATIONS.items():
            t = re.sub(rf"\b{re.escape(abbrev)}\b", expansion, t)
        return t

    def clean_text(self, text: str) -> str:
        """Nettoyage léger : garde ponctuation pour chunking"""
        if not text or not isinstance(text, str):
            return ""
        text = self.expand_medical_abbreviations(text)
        # Retirer URLs/emails seulement
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        # Normaliser espaces multiples
        text = re.sub(r"\s+", " ", text).strip()
        return text


def main():
    default_in = project_root() / "data" / "data_processed" / "ohsumed_cardiovascular.csv"
    default_out = project_root() / "data" / "data_processed" / "ohsumed_cardiovascular_ir.csv"

    ap = argparse.ArgumentParser(description="Preprocessing OHSUMED pour IR/RAG")
    ap.add_argument("--input", default=str(default_in))
    ap.add_argument("--output", default=str(default_out))
    ap.add_argument("--no-expand-abbrev", action="store_true")
    ap.add_argument("--min-text-len", type=int, default=100, help="Longueur minimale texte combiné")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Erreur: {inp}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(inp)

    # Validation
    required = ["title", "abstract"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Erreur: colonnes manquantes: {missing}", file=sys.stderr)
        sys.exit(1)

    # Garantir colonnes standard
    for col in ["seq_id", "split", "category", "mesh_terms"]:
        if col not in df.columns:
            df[col] = ""

    pre = IRPreprocessor(expand_abbreviations=not args.no_expand_abbrev)

    # Texte brut combiné
    df["text_raw"] = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).astype(str)

    # Nettoyage léger
    tqdm.pandas(desc="Preprocessing IR")
    df["text_clean"] = df["text_raw"].progress_apply(pre.clean_text)

    # Filtrer textes trop courts
    df = df[df["text_clean"].str.len() >= args.min_text_len].reset_index(drop=True)

    # Créer ID unique si absent
    if df["seq_id"].isna().any() or (df["seq_id"] == "").any():
        df["doc_id"] = df.index.astype(str)
    else:
        df["doc_id"] = df["seq_id"]

    # Colonnes finales pour IR
    output_cols = ["doc_id", "title", "abstract", "text_clean", "category", "split", "mesh_terms"]
    df = df[output_cols]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n✓ CSV preprocessing IR: {out}")
    print(f"Documents: {len(df)}")
    print(f"Longueur moyenne: {df['text_clean'].str.len().mean():.0f} chars")
    print(f"Vocabulaire unique: {len(set(' '.join(df['text_clean']).split()))}")


if __name__ == "__main__":
    main()