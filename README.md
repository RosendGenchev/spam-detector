Spam Detector (TF-IDF + Naive Bayes)



A simple email/SMS spam detector implemented in Python using a classic machine learning pipeline based on TF-IDF vectorization and Multinomial Naive Bayes classification.



The goal of this project is to demonstrate a clean, well-structured, and installable Python project with proper testing, type hints, and linting, suitable for automated evaluation and live demonstration.



Features



Spam/ham text classification



Robust dataset loading (supports common CSV formats and non-UTF8 encodings)



Train/test split with stratification



Command-line interface (CLI) for live predictions



Unit tests with coverage



Type hints checked with mypy



Code style checked with pylint



Installation



Create and activate a virtual environment, then install the project dependencies using requirements.txt.

On Windows PowerShell, activate the virtual environment and install the dependencies with pip.



Running tests



Run the test suite using pytest.

Test coverage can be measured using pytest-cov.



Type checking and linting



The project uses mypy for static type checking and pylint for code style validation.

Both tools can be executed from the project root against the source package.



Usage



The project provides a command-line interface that loads a trained spam classification model from a local file and allows interactive predictions.

Users can enter text messages and receive a spam or ham classification along with a confidence score.



Notes



The dataset loader supports CSV files with either label/text columns or the common v1/v2 spam dataset format.

Labels are expected to be spam or ham and are handled in a case-insensitive manner.

The project follows a standard src-layout and is compatible with automated grading tools.

