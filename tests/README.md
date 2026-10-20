# MLflow regression tests

Run `python -m unittest discover -s tests -q` from the repository root.
The suite uses standard-library fakes and synthetic fixtures; it does not require
credentials or contact either platform. `seed_mlflow.py`, `extend_mlflow.py`, and
`evaluators.py` are development fixtures, not migration entry points.
