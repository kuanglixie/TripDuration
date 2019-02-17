import numpy as np
import pandas as pd
import datetime as dt
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
from sklearn.utils import shuffle
import lightgbm as lgb

FEAT_ENG = True  # Whether do feature engineering or load from existing dataset
EVAL = True  # True for evaluation, False for predicting on test set
LGB_PARAM = {'objective': 'fair',
             'metric': 'rmse',
             'boosting': 'gbdt',
             'fair_c': 1.5,
             'learning_rate': 0.2,
             'verbose': 0,
             'num_leaves': 60,
             'bagging_fraction': 0.95,
             'bagging_freq': 1,
             'bagging_seed': seed + n,
             'feature_fraction': 0.6,
             'feature_fraction_seed': seed + n,
             'min_data_in_leaf': 10,
             'max_bin': 255,
             'max_depth': 10,
             'reg_lambda': 20,
             'reg_alpha': 20,
             'lambda_l2': 20,
             'num_threads': 30
             }


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def rmsle(predicted, real):
    sum = 0.0
    for x in range(len(predicted)):
        p = predicted[x]
        r = real[x]
        if p < 0:
            p = 0
        if r < 0:
            r = 0
        p = np.log(p + 1)
        r = np.log(r + 1)
        sum = sum + (p - r)**2
    return (sum / len(predicted))**0.5


def bagged_set_cv(X_ts, y_cs, seed, estimators, xt, yt=None):
    baggedpred = np.array([0.0 for d in range(0, xt.shape[0])])
    for n in range(0, estimators):
        d_train = lgb.Dataset(X_ts, np.log1p(y_cs), free_raw_data=False)
        if type(yt) != type(None):
            d_cv = lgb.Dataset(xt, np.log1p(
                yt), free_raw_data=False, reference=d_train)
            model = lgb.train(LGB_PARAM, d_train, num_boost_round=4000,
                              valid_sets=d_cv, verbose_eval=True)
            return model
        else:
            model = lgb.train(LGB_PARAM, d_train, num_boost_round=4000)
        preds = np.expm1(model.predict(xt))
        baggedpred += preds
        print("completed: " + str(n))
    baggedpred /= estimators
    return baggedpred


if FEAT_ENG:
    if EVAL:
        train = pd.read_csv('../../data/train.csv')
        test = pd.read_csv('../../data/test.csv')
    else:
        train = pd.read_csv('../../data/train_eval.csv')
        test = pd.read_csv('../../data/test_eval.csv')
        y_test = np.array(test['duration'].values)

    train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
    test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
    train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
    test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
    train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

    train.loc[:, 'pickup_dt'] = (
        train['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()
    train.loc[:, 'pickup_week_hour'] = train['pickup_weekday'] * \
        24 + train['pickup_hour']

    test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday
    test.loc[:, 'pickup_hour_weekofyear'] = test['pickup_datetime'].dt.weekofyear
    test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour
    test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute
    test.loc[:, 'pickup_dt'] = (
        test['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()
    test.loc[:, 'pickup_week_hour'] = test['pickup_weekday'] * \
        24 + test['pickup_hour']

    train.loc[:, 'pickup_dayofyear'] = train['pickup_datetime'].dt.dayofyear
    test.loc[:, 'pickup_dayofyear'] = test['pickup_datetime'].dt.dayofyear

    train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values,
                                              train['dropoff_latitude'].values, train['dropoff_longitude'].values)

    test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values,
                                             test['dropoff_latitude'].values, test['dropoff_longitude'].values)

    train.loc[:, 'distance_haversine'] = haversine_array(
        train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
    train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(
        train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

    test.loc[:, 'distance_haversine'] = haversine_array(
        test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
    test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(
        test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

    train.loc[:, 'center_latitude'] = (
        train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
    train.loc[:, 'center_longitude'] = (
        train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2
    test.loc[:, 'center_latitude'] = (
        test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
    test.loc[:, 'center_longitude'] = (
        test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2

    coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                        train[['dropoff_latitude', 'dropoff_longitude']].values,
                        test[['pickup_latitude', 'pickup_longitude']].values,
                        test[['dropoff_latitude', 'dropoff_longitude']].values))

    pca = PCA().fit(coords)

    train['pickup_pca0'] = pca.transform(
        train[['pickup_latitude', 'pickup_longitude']])[:, 0]
    train['pickup_pca1'] = pca.transform(
        train[['pickup_latitude', 'pickup_longitude']])[:, 1]
    train['dropoff_pca0'] = pca.transform(
        train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    train['dropoff_pca1'] = pca.transform(
        train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    test['pickup_pca0'] = pca.transform(
        test[['pickup_latitude', 'pickup_longitude']])[:, 0]
    test['pickup_pca1'] = pca.transform(
        test[['pickup_latitude', 'pickup_longitude']])[:, 1]
    test['dropoff_pca0'] = pca.transform(
        test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    test['dropoff_pca1'] = pca.transform(
        test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

    train.loc[:, 'pca_manhattan'] = np.abs(
        train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(train['dropoff_pca0'] - train['pickup_pca0'])
    test.loc[:, 'pca_manhattan'] = np.abs(
        test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(test['dropoff_pca0'] - test['pickup_pca0'])

    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(
        n_clusters=100, batch_size=10000).fit(coords[sample_ind])

    train.loc[:, 'pickup_cluster'] = kmeans.predict(
        train[['pickup_latitude', 'pickup_longitude']])
    train.loc[:, 'dropoff_cluster'] = kmeans.predict(
        train[['dropoff_latitude', 'dropoff_longitude']])
    test.loc[:, 'pickup_cluster'] = kmeans.predict(
        test[['pickup_latitude', 'pickup_longitude']])
    test.loc[:, 'dropoff_cluster'] = kmeans.predict(
        test[['dropoff_latitude', 'dropoff_longitude']])
    t1 = dt.datetime.now()

    train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)

    feature_names = list(train.columns)
    print(np.setdiff1d(train.columns, test.columns))

    do_not_use_for_training = ['id', 'log_trip_duration', 'trip_duration', 'dropoff_datetime', 'pickup_date',
                               'pickup_datetime', 'date']
    feature_names = [
        f for f in train.columns if f not in do_not_use_for_training]
    print('We have %i features.' % len(feature_names))
    print(train[feature_names].count())

    y = np.array(train['duration'].values)
    X = train[feature_names].values
    test_id = np.array(test['id'].values)
    X_test = test[feature_names].values

    if EVAL:
        joblib.dump((X, X_test, y_test, y, test_id), "pikcles_eval.pkl")
    else:
        joblib.dump((X, X_test, y, test_id), "pikcles_train.pkl")
else:
    if EVAL:
        X, X_test, y, y_test, test_id = joblib.load("pikcles.pkl")
    else:
        X, X_test, y, test_id = joblib.load("pikcles.pkl")

print(" final shape of train ",  X.shape)
print(" final shape of X_test ",  X_test.shape)
print(" final shape of y ",  y.shape)


if EVAL:
    seed = 1
    estimators = 1
    model = bagged_set_cv(X, y, seed, estimators, xt=X_test, yt=y_test)
else:
    seed = 1
    path = ''
    outset = "1"
    estimators = 30

    preds = bagged_set_cv(X, y, seed, estimators, X_test, yt=None)
    preds = np.array(preds)

    print("Write results...")
    output_file = "prediction_" + outset + ".csv"
    print("Writing submission to %s" % output_file)
    f = open(output_file, "w")
    f.write("id,trip_duration\n")
    for g in range(0, len(preds)):
        pr = preds[g]
        f.write("%s,%f\n" % (((test_id[g]), pr)))
    f.close()
    print("Done.")
