"""
Vérification + extraction du dataset OHSUMED (ohsumed-all-docs)

Attendu:
- <projet>/data/data_raw/ohsumed-all-docs.tar.gz (téléchargé manuellement)

Action:
- extrait l'archive dans <projet>/data/data_raw/
"""

import sys
import tarfile
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    data_raw = project_root() / "data" / "data_raw"
    archive = data_raw / "ohsumed-all-docs.tar.gz"

    if not archive.exists():
        print(
            "ERREUR: archive manquante.\n"
            "Télécharge manuellement:\n"
            "https://disi.unitn.it/moschitti/corpora/ohsumed-all-docs.tar.gz\n"
            f"et place-la ici:\n{archive}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Vérifier si déjà extrait
    extracted_dirs = [p for p in data_raw.iterdir() if p.is_dir()]
    if extracted_dirs:
        print("✓ OHSUMED déjà extrait.")
        return

    print("Extraction OHSUMED...")
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(path=data_raw, filter="data")
        except TypeError:
            tar.extractall(path=data_raw)

    print("✓ Extraction terminée.")


if __name__ == "__main__":
    main()
