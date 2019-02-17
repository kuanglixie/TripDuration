New York Taxi Trip Duration
==============================

predict New York taxi trip duration

Please check `notebooks` and `src` folder for EDA and model building source code.

Summary of Strategy
------------
what you considered and why, potential improve-
ments, ideas for a real-time implementation, or anything else that you think is relevant to understanding your solution. 


Distance: direct distance distConsin, manhatton distance, fastest route distance
Speed: local traffic: location + time, cluster of location and time, traffic heatmap


Error metics
------------

Please briefly discuss the error metric(s) you believe should be used when travel times are
used to make dispatch and fare decisions.

RMSLE


Ideas of Real-time implementation
------------
The lightGBM model can be train weekly. We can have two levels of predictions. First level is faster, less accurate, and user see this number first. For this level, we can use lightGBM/XGBoost to create smaller set of features (i.e. function like `xgb.create.features`), and then train another linear model on top of those features. The smaller set of features take much less of space and the linear model can be very efficient in the production. For the second level, we can predict on our full models and push to user a few seconds after.


Further improvement
------------

OSRM



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
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
