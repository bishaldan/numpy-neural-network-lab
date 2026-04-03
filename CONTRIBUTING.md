# Contributing

## Local workflow

Use Docker for the main workflow:

```bash
docker compose build
docker compose run --rm trainer
docker compose run --rm test
docker compose run --rm experiments
docker compose up demo
```

## Local Python workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python train.py --quiet
```

## Notes

- Keep the NumPy model as the project centerpiece.
- Prefer adding new reporting or workflow layers around the model instead of replacing it with a framework.
- Update README commands if Docker services or entrypoints change.
