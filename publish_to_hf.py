"""Publica solo el artefacto JSON generado en un Space estático de Hugging Face."""
import os
from pathlib import Path
from huggingface_hub import HfApi

root = Path(__file__).resolve().parent
file_path = root / "data" / "observatorio.json"
token = os.getenv("HF_TOKEN", "").strip()
repo_id = os.getenv("HF_SPACE_ID", "").strip()
if not token:
    raise SystemExit("Falta el secreto HF_TOKEN en GitHub Actions.")
if not repo_id or "/" not in repo_id:
    raise SystemExit("Define HF_SPACE_ID con formato usuario/nombre-del-space.")
if not file_path.exists():
    raise SystemExit(f"No existe el JSON producido: {file_path}")
api = HfApi(token=token)
api.upload_file(
    path_or_fileobj=str(file_path),
    path_in_repo="data/observatorio.json",
    repo_id=repo_id,
    repo_type="space",
    commit_message="Actualizar panel con fuentes reales DANE SIPSA-P y NASA POWER",
)
print(f"Publicado data/observatorio.json en {repo_id}")
