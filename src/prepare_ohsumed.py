"""
Parsing + extraction du sous-corpus cardiovasculaire depuis ohsumed-all-docs (Moschitti)

Entrée:
- <projet>/data/data_raw/ (après download_ohsumed.py)

Sortie:
- <projet>/data/data_processed/ohsumed_cardiovascular.csv
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
from tqdm import tqdm


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# =========================
# Format Moschitti (training/test + catégories type C14)
# =========================

def find_moschitti_c14_dirs(root: Path, label: str = "C14") -> List[Path]:
    """
    Retourne liste de répertoires contenant le label (ex: C14).
    Supporte:
    - ohsumed-all/C14/
    - training/C14/ et test/C14/
    - ohsumed-first-20000-docs/training/C14/
    """
    candidates = []

    # Pattern 1: ohsumed-all/CXX/ ou tout répertoire CXX
    for d in root.rglob(label):
        if d.is_dir():
            candidates.append(d)

    # Pattern 2: sous training/test explicite
    for split in ["training", "train", "test", "testing"]:
        for split_dir in root.glob(f"**/{split}"):
            if not split_dir.is_dir():
                continue
            for subcat in split_dir.iterdir():
                if subcat.is_dir() and label in subcat.name:
                    candidates.append(subcat)

    return sorted(set(candidates))


def parse_simple_doc_file(filepath: Path) -> Optional[Dict[str, str]]:
    """
    Format simple:
    - ligne 1 = titre
    - reste = abstract
    """
    try:
        txt = filepath.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            return None
        lines = txt.split("\n", 1)
        title = lines[0].strip()
        abstract = lines[1].strip() if len(lines) > 1 else ""
        return {"title": title, "abstract": abstract}
    except Exception:
        return None


def load_moschitti_c14(root: Path, label: str = "C14") -> pd.DataFrame:
    print(f"\n=== Recherche format Moschitti ({label}) ===")
    label_dirs = find_moschitti_c14_dirs(root, label=label)
    
    print(f"Répertoires {label} trouvés: {len(label_dirs)}")
    for ld in label_dirs:
        print(f"  {ld.relative_to(root)}")
    
    if not label_dirs:
        raise FileNotFoundError(
            f"Format Moschitti non détecté: aucun répertoire {label}.\n"
            f"Arborescence attendue: data_raw/.../ohsumed-all/{label}/ ou training/{label}/"
        )

    rows: List[Dict[str, str]] = []
    total = 0

    for label_dir in label_dirs:
        # Détecter split depuis le chemin
        parts = [p.lower() for p in label_dir.parts]
        if "training" in parts or "train" in parts:
            split = "training"
        elif "test" in parts or "testing" in parts:
            split = "test"
        else:
            split = "all"  # Pas de split explicite
        
        files = [p for p in label_dir.rglob("*") if p.is_file()]
        print(f"Répertoire {label_dir.name} (split={split}): {len(files)} fichiers")
        
        for fp in tqdm(files, desc=f"Parsing {split}/{label}", leave=False):
            total += 1
            doc = parse_simple_doc_file(fp)
            if not doc:
                continue
            
            doc_id = f"{split}:{fp.relative_to(root)}"
            rows.append(
                {
                    "seq_id": doc_id,
                    "medline_ui": "",
                    "mesh_terms": label,
                    "title": doc["title"],
                    "abstract": doc["abstract"],
                    "source": "",
                    "authors": "",
                    "split": split,
                    "category": label,
                }
            )

    df = pd.DataFrame(rows)
    print(f"Docs lus (brut): {total}")
    print(f"Docs gardés (non vides): {len(df)}")
    return df


# =========================
# Fallback ancien format OHSUMED (.I/.M)
# =========================

def find_ohsumed_standard_dir(root: Path) -> Path:
    """Cherche fichiers ohsumed.* avec patterns alternatifs."""
    # Pattern 1: ohsumed.* direct
    if list(root.glob("ohsumed.*")):
        return root
    
    # Pattern 2: sous-répertoires profonds
    for d in root.rglob("*"):
        if d.is_dir() and list(d.glob("ohsumed.*")):
            return d
    
    # Pattern 3: fichiers .txt ou sans extension dans training/test
    training_test = list(root.glob("**/training")) + list(root.glob("**/test"))
    if training_test:
        return training_test[0].parent
    
    # Pattern 4: recherche de fichiers contenant .I\s+\d+ (format standard)
    for f in root.rglob("*"):
        if f.is_file() and f.stat().st_size < 100_000_000:  # < 100MB
            try:
                sample = f.read_text(encoding="latin-1", errors="ignore")[:5000]
                if re.search(r"\.I\s+\d+", sample):
                    print(f"Format standard détecté: {f}")
                    return f.parent
            except Exception:
                continue
    
    # Diagnostic détaillé
    print("\n=== DIAGNOSTIC STRUCTURE ===")
    print(f"Racine: {root}")
    files = sorted(root.rglob("*"))[:100]
    print(f"Fichiers/dossiers présents ({len(files)} premiers):")
    for p in files:
        typ = "[DIR]" if p.is_dir() else f"[{p.suffix or 'NO_EXT'}]"
        print(f"  {typ} {p.relative_to(root)}")
    
    raise FileNotFoundError(
        f"Aucun format OHSUMED reconnu dans {root}.\n"
        "Formats attendus:\n"
        "  - Moschitti: training/C14/ et test/C14/\n"
        "  - Standard: fichiers ohsumed.* avec format .I/.M\n"
        "Vérifiez l'extraction de l'archive."
    )


def parse_document(doc_text: str) -> Optional[Dict[str, str]]:
    doc: Dict[str, str] = {}
    m = re.search(r"\.I\s+(\d+)", doc_text)
    if not m:
        return None
    doc["seq_id"] = m.group(1)

    def grab(tag: str) -> str:
        m2 = re.search(rf"\.{tag}\s*\n(.+?)(?=\n\.|\Z)", doc_text, re.DOTALL)
        return m2.group(1).strip() if m2 else ""

    doc["medline_ui"] = grab("U")
    doc["mesh_terms"] = grab("M")
    doc["title"] = grab("T")
    doc["abstract"] = grab("W")
    doc["source"] = grab("S")
    doc["authors"] = grab("A")
    return doc


def is_mesh_prefix(mesh_terms: str, prefix: str = "C14") -> bool:
    if not mesh_terms:
        return False
    return bool(re.search(rf"\b{re.escape(prefix)}\.\d+", mesh_terms))


def iter_documents_from_file(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="latin-1", errors="ignore")
    except Exception:
        text = path.read_text(encoding="utf-8", errors="ignore")
    docs = re.split(r"(?=\.I\s+\d+)", text)
    return [d for d in docs if d.strip()]


def load_standard_c14(root: Path, mesh_prefix: str = "C14") -> pd.DataFrame:
    corpus_dir = find_ohsumed_standard_dir(root)
    oh_files = sorted(corpus_dir.glob("ohsumed.*"))

    all_docs: List[Dict[str, str]] = []
    total = 0

    for f in tqdm(oh_files, desc="Parsing ohsumed.*"):
        for doc_text in iter_documents_from_file(f):
            total += 1
            doc = parse_document(doc_text)
            if not doc:
                continue
            if is_mesh_prefix(doc.get("mesh_terms", ""), mesh_prefix):
                all_docs.append(doc)

    print(f"Docs parsés: {total}")
    print(f"Docs gardés ({mesh_prefix}.*): {len(all_docs)}")
    return pd.DataFrame(all_docs)


# =========================
# Main
# =========================

def main():
    default_raw = project_root() / "data" / "data_raw"
    default_out = project_root() / "data" / "data_processed" / "ohsumed_cardiovascular.csv"

    ap = argparse.ArgumentParser(description="Extraire le sous-corpus cardiovasculaire (C14)")
    ap.add_argument("--data-dir", default=str(default_raw), help="Dossier raw (défaut: <projet>/data/data_raw)")
    ap.add_argument("--label", default="C14", help="Catégorie à extraire pour le format Moschitti (défaut: C14)")
    ap.add_argument("--min-abstract-len", type=int, default=50, help="Longueur minimale d'abstract (défaut: 50)")
    ap.add_argument("--output", default=str(default_out), help="CSV de sortie")
    args = ap.parse_args()

    raw_root = Path(args.data_dir)

    # 1) Moschitti d'abord
    try:
        df = load_moschitti_c14(raw_root, label=args.label)
        mode = "moschitti"
    except Exception as e:
        print(f"⚠ Format Moschitti non utilisé ({e}). Fallback format standard .I/.M")
        df = load_standard_c14(raw_root, mesh_prefix=args.label)
        mode = "standard"

    if df.empty:
        raise ValueError("Aucun document chargé.")

    df["title"] = df.get("title", "").fillna("").astype(str)
    df["abstract"] = df.get("abstract", "").fillna("").astype(str)
    df = df[df["abstract"].str.len() >= args.min_abstract_len].reset_index(drop=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Mode: {mode}")
    print(f"CSV écrit: {out}")
    print(f"Docs finaux: {len(df)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr)
        sys.exit(1)