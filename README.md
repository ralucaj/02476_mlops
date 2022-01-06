mlops
==============================

MLOps course project

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

Requirements prep:
```
make requirements  # install everything in the requirements.py file
```

Data prep:
```
 make data  # runs the make_dataset.py file
 make clean  # clean __pycache__ files
```
Reads the `.npz` files in `data/raw` and transforms them into pytorch tensors in `data/processed`. 

Train model: 
```
make train
```
Trains the model using the data from `data/processed/train.pt` using the architecture defined in `src/models/model.py`.
Saves the trained model in `models/model.pth` and the training loss curve in `reports/figures/training_loss.png`.

Predict:
```
python src/models/predict_model.py \
     models/model.pt \  # file containing a pretrained model
     data/example_images.npy
```
Saves the class prediction for each image in `./predictions.pt`.

Visualize:
```
 python src/visualization/visualize.py "./models/model.pth"
```
Creates a T-SNE visualization of the last feature layer for the test set in `./figures/tsne.png`

#### Wandb
Report available [here](https://wandb.ai/ralucaj/02476_mlops-src_models/reports/Untitled-Report--VmlldzoxNDEwMzc5?accessToken=latws5d49t58l6rszu8e32k3kd3gky5swnadu7gn9h1q7j8ieblv95g9x112kt28)