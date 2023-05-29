import sys
from pathlib import Path

print(sys.argv)
project_name = sys.argv[1]

project_path = Path(project_name)
project_path.mkdir(parents=True, exist_ok=True)

(project_path / "predicted_data").mkdir(parents=True, exist_ok=True)
(project_path / "training_data").mkdir(parents=True, exist_ok=True)
(project_path / "main.py").touch(exist_ok=True)
