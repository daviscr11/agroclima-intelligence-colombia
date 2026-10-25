"""Publica la interfaz estática y los artefactos que consume en Hugging Face."""
import os
from pathlib import Path
from huggingface_hub import HfApi

root = Path(__file__).resolve().parent
token = os.getenv("HF_TOKEN", "").strip()
repo_id = os.getenv("HF_SPACE_ID", "").strip()
if not token:
    raise SystemExit("Falta el secreto HF_TOKEN en GitHub Actions.")
if not repo_id or "/" not in repo_id:
    raise SystemExit("Define HF_SPACE_ID con formato usuario/nombre-del-space.")

# La página del Space carga todos estos archivos en el navegador. Publicarlos en
# una sola operación evita dejar el HTML apuntando a datos que aún no existen.
files = [
    "README.md",
    "index.html",
    "data/observatorio.json",
    "data/residual_moran_panel_audit.csv",
    "data/colombia_outline.geojson",
    "data/colombia_departments.geojson",
]
missing = [name for name in files if not (root / name).is_file()]
if missing:
    raise SystemExit("Faltan archivos requeridos para el Space: " + ", ".join(missing))

api = HfApi(token=token)
api.upload_folder(
    folder_path=str(root),
    repo_id=repo_id,
    repo_type="space",
    allow_patterns=files,
    commit_message="Actualizar interfaz, residuos y mapas del observatorio",
)
print(f"Publicados {len(files)} archivos del observatorio en {repo_id}")
