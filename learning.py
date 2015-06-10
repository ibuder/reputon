# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:08:51 2015

@author: ibuder
"""

import pandas as pd
import sklearn as skl  # FIXME what's the standard?
import sklearn.cross_validation

import db

# TODO use OneHotEncoder to process categorical cols


def make_features1(listings):
    """
    Get features for ML from DataFrame

    May return a view (not copy)
    This is feature set version 1 (other versions may exist)
    """
    # TODO think more about whether to use nReviews
    # FIXME may need to apply standard scaler
    # FIXME need to convert categorical columns to multiple binary columns
    # FIXME how to deal with missing data (e.g. -1 in some columns)
    # FIXME add length of heading and description and photo comments
    use_columns = [column for column in listings.columns if 'amenity' in column]
    use_columns.extend(('bathrooms', 'bedrooms', 'beds', 'instantBookable',
                        'isCalAvailable', 'lastUpdatedAt', 'occupancy',
                        'propType', 'roomType', 'securityDeposit', 'size',
                        'extraGuestsFee', 'extraGuestsAfter', 'fee0',
                        'lat', 'lng', 'nPhotos', 'maxNight', 'minNight',
                        'monthly', 'nightly', 'weekend', 'weekly',
                        'nReviews',))
    return listings[use_columns]


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
    return skl.dummy.DummyClassifier(strategy='stratified')

if __name__ == '__main__':
    engine = db.create_root_engine()
    rawtable = pd.io.sql.read_sql_table('listings', engine, index_col='id')
    frame_out = make_features1(rawtable)
