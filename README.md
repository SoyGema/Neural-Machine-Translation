Neural Machine Translation
==============================

The Challenge
------------

Trains a model that transforms a language **text from one language into another**, taking into account **LLM fundamentals:** Transformers architecture and feature engineering coming from Natural Language Processing.

**Why is this suitable/interesting for DVC ? and VSCode DVC extension?**

- [DVC](https://dvc.org/) allows us to **version 9 different language** datasets to be trained.

- [DVC Pipelines](https://dvc.org/doc/user-guide/pipelines/defining-pipelines) It allows us to **train transformer architecture for each language avoiding code duplication and controlling versioning** by language in datasets, feature engineering parameters and architecture variations.

- **VSCode [DVC extension](https://marketplace.visualstudio.com/items?itemName=Iterative.dvc) table and plots** allow us to benchmark how well the same/best feature engineering and the same/best architecture perform with various languages and visualize learning and attention heads.

What is Neural Machine Translation?
------------

Neural Machine Translation’s main goal is to transform a sequence from one language into another sequence to another one. It is an approach to machine translation inside NLP that uses Artificial Neural Networks to predict the likelihood of a sequence of words, often trained in an end-to-end fashion and can generalize well to very long word sequences. Formally it can be defined as a NN that models the conditional probability $ p(y|x)$ of translating a sentence $x1...xn$ into $y1...yn$.

Why Transformers for Neural Machine Translation?
------------

Transformer has been widely adopted in Neural Machine Translation (NMT) because of its large capacity and parallel training of sequence generation. However, the deployment of Transformers is challenging because different scenarios require models of different complexities and scales.

Current state of the project
------------

The Project structure divides as follows.
Tokenizer language has created 9 datasets of 9 tokenized languages following the word embeddings tutorial. This is separated from the Neural Machine Translation project for faster integration. These datasets are integrated with DVC.
In ´src/features´ you can see feature engineering steps, that are related to the feature engineering transformation

* **load_dataset.py** Loads data 
* **tokenizer_transformer.py**  Tokenize the dataset and makes batches
* **positional_encoding.py** Makes the embeddings

In ´src/models´ you can find the modules for training and for inference

* **train_transformer.py** Trains the transformer, declaring the arguments for encoder and decoder modules

In ´src/visualization´ you can find the visualizations for VS Code extension

* **metrics.py** define the loss_function and accuracy_function.
* **visualize.py** define the attention heads that will be plotted in visual studio.

The current tasks to do include
- [ ] debugging Inference and saving model.
- [ ] DVC pipeline. Currently creating ´dvc.yaml´ and params.yml file 
- [ ] integration with DVC VsCode Extension. 


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





@inproceedings{Ye2018WordEmbeddings,
  author  = {Ye, Qi and Devendra, Sachan and Matthieu, Felix and Sarguna, Padmanabhan and Graham, Neubig},
  title   = {When and Why are pre-trained word embeddings useful for Neural Machine Translation},
  booktitle = {HLT-NAACL},
  year    = {2018},
  }