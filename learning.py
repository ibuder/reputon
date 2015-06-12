# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:08:51 2015

@author: ibuder
"""

import pandas as pd
import sklearn as skl  # FIXME what's the standard?
import sklearn.cross_validation
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import numpy as np
import matplotlib.pyplot as plt

import db

# TODO Think about having binary features for no bathroom, bedroom, etc. info
#   i.e. when those columns are -1
#   non-linear models should be ok
#   (e.g. random forest can have special case for -1)
# FIXME how to deal with missing data (e.g. -1 in some columns)

# TODO map checkIn, checkOut to numerical
# TODO bag of words for text


def make_features1(listings):
    """
    Get features for ML from DataFrame

    May return a view (not copy)
    This is feature set version 1 (other versions may exist)
    """

    use_columns = [column for
                   column in listings.columns if 'amenity' in column]
    use_columns.extend(('bathrooms', 'bedrooms', 'beds', 'instantBookable',
                        'isCalAvailable', 'lastUpdatedAt', 'occupancy',
                        'propType', 'roomType', 'securityDeposit', 'size',
                        'extraGuestsFee', 'extraGuestsAfter', 'fee0',
                        'lat', 'lng', 'nPhotos', 'maxNight', 'minNight',
                        'monthly', 'nightly', 'weekend', 'weekly',
                        'nReviews',))
    return listings[use_columns]


def make_features2(listings):
    """
    Get features for ML from DataFrame

    May return a view (not copy)
    This is feature set version 2 (other versions may exist)
    Includes categorical features as multiple binary features
    Includes lengths of listing text
    """
    features1 = make_features1(listings)
    # Categorical features
    # convert categorical columns to multiple binary columns
    cancellation_values = {'Flexible': 0, 'Moderate': 1, 'Strict': 2,
                           'Super Strict': 3}
    response_time_values = {'': 0, 'a few days or more': 1, 'within a day': 2,
                            'within a few hours': 3, 'within an hour': 4}
    # dict.get maps keys to values, so map will return array of ints
    cancellation = listings.cancellation.map(cancellation_values.get)
    responseTime = listings.responseTime.map(response_time_values.get)
    # listings.propType  # categorical with 0--4 so far seen
    # listings.roomType  # categorical with 0--2 so far seen
    categorical_features = pd.concat((cancellation, responseTime,
                                      listings.propType, listings.roomType,),
                                      axis=1)
    # Hardcode number of values so result will have same number of columns
    #   even if not all possible values are present (e.g. encoding 1 row)
    encoder = sklearn.preprocessing.OneHotEncoder(n_values=(4, 5, 5, 3,),
                                                  categorical_features='all',
                                                  sparse=False,
                                                  handle_unknown='error')
    coded_features = pd.DataFrame(encoder.fit_transform(categorical_features),
                                  index=categorical_features.index)
    coded_features.rename(
        columns=lambda name: 'features2_categorical' + str(name),
        inplace=True)

    # Text lengths
    descriptionLength = listings.description.map(len)
    headingLength = listings.heading.map(len)
    photosCommentsLen = listings.photosComments.map(len)

    result = pd.concat((features1, descriptionLength, headingLength,
                        photosCommentsLen, coded_features,), axis=1, copy=True)
    result.rename(columns={'description': 'descriptionLength',
                           'heading': 'headingLength',
                           'photosComments': 'photosCommentsLength'},
                           inplace=True)
    # remove categorical features that were in features1 and
    #   are now in coded_features
    del result['propType']
    del result['roomType']
    return result


def make_features3(listings):
    """
    Get features for ML from DataFrame

    May return a view (not copy).
    This is feature set version 3 (other versions may exist).
    Does not include # of reviews.
    Includes binary features for top 50 words in heading
    """
    result = make_features2(listings)
    # Top 50 words found in headings in SF (see ML_2015-06-11 iPython notebook)
    heading_words = ['in', 'Mission', 'Room', 'Private', 'the', 'SF',
       'Victorian', 'with',
       'Studio', 'of', '1', '&', 'Sunny', 'Hill', '-', 'Modern', 'Apartment',
       'room', '2', 'Bedroom', 'Home', 'Cozy', 'Flat', 'Beautiful', 'Apt',
       'Valley', 'San', 'Charming', 'Garden', 'Spacious', 'Park', 'Heights',
       'Francisco', 'View', 'Noe', 'bedroom', 'House', 'near', 'Large', 'and',
       'to', 'Castro', 'Nob', 'Heart', 'Beach', 'Great', 'BR', 'Haight',
       'Luxury', 'private']
    for word in heading_words:
        result[word + '_in_heading'] = listings.heading.map(lambda heading:
                                                            word in heading)

    # Bad idea to use number of reviews: since it doesn't help in the use case
    #   (nReviews is always 0)
    #   it makes the generalization accuracy look better than it should be
    del result['nReviews']
    return result


def categorize_rating(rating):
    """
    Map continuous rating to discrete categories
    """
    if rating > 4.5:
        return '5-'
    elif rating > 4:
        return '4.5-'
    else:
        return '4-'
    assert 0  # unexpected rating


def get_training_test_set(listings, make_features=make_features1):
    """
    Get training/testing set split for ML

    Returns (Xtrain, Xtest, ytrain, ytest)

    listings: DataFrame from SQL table listings

    make_features: function that maps listings to features
    """
    labeled_listings = listings.dropna(subset=('rating',))
    X = make_features(labeled_listings)
    y = labeled_listings.rating.map(categorize_rating)
    return skl.cross_validation.train_test_split(X, y)


def get_logistic_regression_clf1():
    """
    Logistic Regression with L2 regularization and feature normalization

    Parameters not optimized at all
    """
    clf = skl.linear_model.LogisticRegression(multi_class='ovr', C=0.010)
    return skl.pipeline.Pipeline([
        ('scaler', skl.preprocessing.StandardScaler()), ('logistic', clf)])


def get_dummy_clf():
    """
    Dummy classifier for comparison
    """
    # Most frequent is the best dummy when using accuracy metric
    return skl.dummy.DummyClassifier(strategy='most_frequent')


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring='accuracy', dummy=None):
    """
    Calculate and plot learning curves

    dummy: A dummy model to compare to
    """

    alpha = 0.1
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    n_train, train_scores, test_scores = skl.learning_curve.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    if dummy:
        (dummy_n_train, dummy_train_scores,
         dummy_test_scores) = skl.learning_curve.learning_curve(
         dummy, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
         scoring=scoring)
        dummy_test_scores_mean = np.mean(dummy_test_scores, axis=1)
        dummy_test_scores_std = np.std(dummy_test_scores, axis=1)
    plt.grid()
    if dummy:
        plt.fill_between(n_train,
                         dummy_test_scores_mean - dummy_test_scores_std,
                         dummy_test_scores_mean + dummy_test_scores_std,
                         alpha=alpha, color='b')
        plt.plot(n_train, dummy_test_scores_mean, 'o-', color='b',
                 label="Dummy model")
    plt.fill_between(n_train, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(n_train, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(n_train, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(n_train, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.grid(True, which='both', axis='y')
    return plt


if __name__ == '__main__':
    engine = db.create_root_engine()
    rawtable = pd.io.sql.read_sql_table('listings', engine, index_col='id')
    frame_out = make_features3(rawtable)
