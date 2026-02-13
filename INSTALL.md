\# Installation and Usage



\## Requirements

\- Python 3.10 or newer

\- Tested on Windows with Python 3.11



\## Setup (Windows PowerShell)



Clone the repository:

git clone https://github.com/RosendGenchev/spam-detector.git

cd spam-detector



Create and activate virtual environment:

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1



Install dependencies:

pip install -U pip

pip install -r requirements.txt

pip install -e .



\## Train the model

python -m spam\_detector.train



This will train the model and save it locally in the model/ directory.



\## Run CLI demo

python -m spam\_detector.cli



Enter a message to classify it as spam or ham.


\## Evaluate + generate plots

python -m spam\_detector.evaluate

This generates plots in reports/ (confusion matrix, model comparison).


\## Run API (FastAPI)

uvicorn spam\_detector.api:app --reload

POST JSON to /predict, example body:

{"text": "free prize, click now"}



\## Run tests

pytest -q



Run tests with coverage:

pytest --cov=spam\_detector --cov-report=term-missing



\## Static analysis

Run type checking:

mypy src



Run linting:

pylint src



