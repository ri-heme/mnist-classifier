MNIST Classifier
==============================

MNIST classifier to work and test out the exercises and learnings from the [Machine Learning Operations course](https://skaftenicki.github.io/dtu_mlops/) @ DTU. This classifier integrates libraries and concepts from the ML Toolbox, such as code organization (cookiecutter), styling (flake8, black, isort), CI (GitHub Actions), boilerplates (PyTorch Lightning), reproducibility (Hydra, Docker), and logging/profiling (Weights & Biases), scalability, among others.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── conf
    │   ├── main.yaml      <- Main default configuration file.
    │   │
    │   ├── experiment     <- Experiment overrides to any default configuration.
    │   ├── model          <- Default model configuration.
    │   └── training       <- Default training loop configuration.
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── train_model.py
    │
    └── pyproject.toml     <- configuration for black


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
