New York Taxi Trip Duration
==============================

Please check `notebooks` and `src` folder for EDA and model building source code.

Summary of Strategy:
------------
The most important element would affect the trip duration is the geo distance between the pick-up location and drop-off location. There are several major ways to calculate the distance, including shortest distance, manhatton distance, fastest route distance. The shortest distance between two points on a spherical earth is calculated using distCosine function of the geosphere package for spherical trigonometry. I also use the manhatton distance, which is the sum of horizontal and vertical distance between two points. For the fastest route, I found a very useful software OSRM (Open Source Routing Machine).  

However, the relationship between trip duration and distance is not linear(as shown by plots in EDA). This lead to another important factor, the travel speed. Compared to the distance, the speed is more difficult to predict. I identified the features that would affect the speed. Many features related to the pick-up time and drop-off time, like hour of the day and day of week. There are also features related to the geo location, like the start_lat and start_lng. These nearby traffic features might have more predictive power if we take both time dimension and geo dimension into consideration. I cluster trips that happen within certain geo and time intervals.

For the details of my strategies, you can read `notebooks/01_EDA.nb.html` , which include data wrangling, exploratory data analysis and feature engineering.

I use lightgbm package which implements Gradient Boosting Tree. There several reasons for this selection. Firstly, we want to capture the interactions between features and the nonlinear relationship between features and target variable. Secondly, we have many categorical variables, and tree-based models have advantage to deal with them. Third, since I only have 24 hours for this project, I need to choose a model that is fast on a big data set. The code is located at `src/models/train_model.py`.


Major Results:
-------------
We have save 25% of training set as evaluation set.
Mean squared error (evaluation set):104006.2
Rooted mean squared error (training set): 322.5
We have output file, `prediction_1.csv`.


Potential Improvement:
------------------------

Data from OSRM provide a much more accurate distance and duration estimation between start and end location, if no other traffic condition.  The `duration` and `distance` features got from OSRM is very promising and can serve as great features for trip duration prediction. However, for 1000 samples, the program will take ~3 mins to run on my laptop and my home Internet. The total running time for all samples will be more than 24 hours.

After profiling the code, I found that the main bottleneck here is the http request to OSRM service (the `RCurl` call). If I have more time, I would write a distributed code to send request using multiple threads and deploy to several small size AWS (Amazon Web Service) instances. I would guess OSRM service will have some limitation for each IP address, so I will also need to tune the number of request/s per IP. However, for a real-time service, we need to have our own service for fast route calculation. For further detail, you can read `notebooks/02_OSRM.Rmd`

To improve the prediction of travel speed, I think creating traffic heatmaps would be a good idea.  

The traffic hotness of a certain area might differ across hours and day of a week, so we need to create several heatmaps. For each trip, we can extract the coordinates longitude and latitude that declinate the fastest route. Then, I can randomly select 10 samples per mile to represent the trip route, and use geohash to encode the selected coordinates. For each hashed area in the map, I can count the number of coordinates which represent the hotness of that area.

Then, given the traffic heatmap, we can identify the hotness of each trip. For this trip, I randomly select 10 encoded coordinates per mile along the fastest route to delineate the trip. We can count the number of coordinates for each hashed area. Then, we can use the weighted average to calculate the traffic hotness of the trip.  

In addition, I can use data from other sources to improve the performance, like information on weather, holiday, and events.


Ideas of Real-time implementation
------------
The lightGBM model can be train weekly. We can have a two-level prediction. First level is faster, less accurate, and user see this number first. For this level, we can use lightGBM/XGBoost to create smaller set of features (i.e. function like `xgb.create.features`), and then train another linear model on top of those features. The smaller set of features take much less of space and the linear model can be very efficient in the production. For the second level, we can predict on our full models and push to user a few seconds after.


Error metrics
------------

I think we should choose Root Mean Squared Logarithmic Error as the error metric(s) when travel times are used to make dispatch and fare decisions. There are two reasons. Firstly, users can tolerate bigger errors when they have longer trip. Using this measurement we give the error less punishment when the trip is longer. Secondly, most of our trips are less than 20 minutes, so we want our model to focus on these trips instead of being distorted by rare long trips.   

 Project Organization
--------------------

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
