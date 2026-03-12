import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
project_name = "mental_health"
list_of_files = [
    "github/workspace/.gitkeep",
    f"src/{project_name}/__init__.py",
    # Components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    # Pipelines
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/train_pipeline.py",
    f"src/{project_name}/pipelines/predict_pipeline.py"
    # Utils
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    # core
    f"src/{project_name}/logger.py",
    f"src/{project_name}/exception.py",
    # Config
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    # Artifacts
    "artifacts/.gitkeep"
    # Root files
    "app.py",
    "requirements.txt",
    "setup.py",
    "Readme.md"
]
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")