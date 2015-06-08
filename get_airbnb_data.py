# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 06:36:06 2015

@author: ibuder
"""

import requests
import pandas as pd
import numpy as np
import db


# FIXME Clean up all dataframe dtypes
def make_airbnb_json_dataframe(json, n_amenities=50):
    """
    Turn API JSON output into DataFrame

    json should be the top level object returned be requests.get().json()
    n_amenities: Maximum number of amenity types
    """

    rawframe = pd.DataFrame(json['result'])
    if rawframe.empty:
        return None
    # The next many lines munge rawframe into something that can go to SQL

    amenities_lists = rawframe.amenities.map(lambda amens:
                                             [amen['id'] for amen in amens])
    amenities = pd.DataFrame()
    # loop over amenities and create new columns with map
    for iamen in range(n_amenities):
        amenities['amenity' + str(iamen)] = amenities_lists.map(
            lambda amenities_list: iamen in amenities_list)
    attr = rawframe.attr.apply(lambda attr_dict: pd.Series(attr_dict))
    attr.cancellation = attr.cancellation.map(
        lambda cancel_dict: cancel_dict['text'])
    attr['extraGuestsFee'] = attr.extraGuests.map(
        lambda guest_dict: guest_dict['fee'])
    attr['extraGuestsAfter'] = attr.extraGuests.map(
        lambda guest_dict: guest_dict['after'])
    del attr['extraGuests']

    def extract_fee(fee):
        if fee:
            # Expected fee is a list containing dicts
            return fee[0]['fee']
        else:
            return 0
    # Might find more types of fees later, so call this one fee0
    attr['fee0'] = attr.fees.map(extract_fee)
    del attr['fees']

    def get_id(dict_):
        return dict_['id']
    attr.propType = attr.propType.map(get_id)
    attr.roomType = attr.roomType.map(get_id)
    latLng = rawframe.latLng.apply(pd.Series).rename(
        columns={0: 'lat', 1: 'lng'})
    location = rawframe.location.apply(pd.Series)
    del location['all']
    photos = pd.DataFrame(rawframe.photos.map(len)).rename(
        columns={'photos': 'nPhotos'})
    photos['photosComments'] = rawframe.photos.map(lambda photo_list:
                ' '.join([photo['caption'] for photo in photo_list]))
    prices = rawframe.price.apply(pd.Series)
    reviews = rawframe.reviews.apply(pd.Series)
    reviews.rename(columns={'count': 'nReviews'}, inplace=True)

    # This info won't be available for unreviewed listings
    del reviews['entries']
    reviews.loc[reviews.rating < 0, 'rating'] = np.nan
    return pd.concat((amenities, attr, rawframe.id, latLng, location, photos,
                      prices, reviews,), axis=1)


def get_airbnb_json(params={}):
    """
    Use API to get JSON objects for a bunch of airbnb listings.

    Returns request.json()
    """
    headers = {
        "X-Mashape-Key": FIXME put in gitignored settings file,
        "Accept": "application/json"}
    params_default = {'latitude': 37.762673, 'longitude': -122.438554,  # SF
          'provider': 'airbnb', 'resultsperpage': 50, 'sort': 'low2high',
          'page': 4, 'maxdistance': 5}
    params_default.update(params)
    params = params_default
    return requests.get('https://zilyo.p.mashape.com/search',
                        headers=headers, params=params).json()
# FIXME
# See http://stackoverflow.com/questions/24879156/pandas-to-sql-with-sqlalchemy-duplicate-entries-error-in-mysqldb
# for help with duplicates
engine = db.create_root_engine()
for page in range(40, 100):
    json = get_airbnb_json({'page': page})
    make_airbnb_json_dataframe(json).to_sql('listingsTest', engine,
                                            if_exists='append', index=False)
