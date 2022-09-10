Neural Machine Translation
==============================


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
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
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>






# Attention-based Neural-Machine-Translation

Neural Machine Translation is an approach to machine translation that uses Artificial
Neural Networks to predict likelihood of a sequence of words, often trained in an 
end-to-end fashion and has the ability to generalize well to very long word sequences.
Formally it can be defined as a NN that models the conditional probability p(y x) of
translating a sentence x1...xn into y1...yn.

# POC 1
Using [transformers for Neural Machine Translation](https://arxiv.org/pdf/2106.02242.pdf)
The high level idea behind transformers is to design the attention mechanism: the ability
to set attention at different parts of a sequence for calculating a sequence representation.
It presents several difference with respect to past sequential architectures.
 
 1. No supositions about temporal/spacial relationship in between data.Therefore, it
 doesn´t think about them about a sequence per se. 
 
 2. Layer outputs can be calcualted in parallel
 
 3. Learn long dependencies 



# Datasets

The project attempts to use the Spanish Dataset.
Other research include
French to English
German to English
Portuguese to English 
Russian to English

# POC 2?

Research has take the attention-based mechanism into this field. The exploration implements
the paper based on two simple and effective classes of attentional mechanism:

* global approach : attends to all source words
* local: looks at one sentence. Conceived as an iteration from the existing previous
research literature. The difference is that it is differenciable almost everywhere, computationally
less expensive. This is based moslty in LSTM architectures.
