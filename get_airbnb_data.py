# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 06:36:06 2015

@author: ibuder
"""

import requests
import pandas as pd
import numpy as np
import re

import settings
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


def airbnb_url_to_id(url):
    m = re.search(r'rooms/([0-9]+)\??', url)
    if m:
        return 'air' + m.group(1)
    return None


mashape_headers = {
        "X-Mashape-Key": settings.mashape_testing,
        "Accept": "application/json"}


def get_airbnb_by_id(id_, params={}):
    """
    Get single airbnb listing with known id

    Returns request.json()
    """
    params_default = {'ids': id_}
    params_default.update(params)
    params = params_default
    return requests.get('https://zilyo.p.mashape.com/search',
                        headers=mashape_headers, params=params).json()


def get_airbnb_json(params={}):
    """
    Use API to get JSON objects for a bunch of airbnb listings.

    Returns request.json()
    """

    params_default = {'latitude': 37.762673, 'longitude': -122.438554,  # SF
          'provider': 'airbnb', 'resultsperpage': 50, 'sort': 'low2high',
          'page': 4, 'maxdistance': 5}
    params_default.update(params)
    params = params_default
    return requests.get('https://zilyo.p.mashape.com/search',
                        headers=mashape_headers, params=params).json()



def populate_db_raster(latitude_range, longitude_range, step_km=1,
                       max_page=50):
    """
    Get listings in a lat/lng box, then put all into database

    Searches at a grid of points to cover the box
    grid points are half-open i.e. [start, stop)

    latitude_range: (start_lat, stop_lat)

    max_page: maximum number of pages to try at a single point
    """
    # Compute the (lat, lon) points to sample
    earth_circumference_km = 6378.1 * 2 * np.pi  # 2pi*radius
    km_to_lat = 360 / earth_circumference_km
    km_to_lon = 360 / (earth_circumference_km *
                       np.cos(np.deg2rad(np.mean(latitude_range))))
    lat_points = np.arange(*latitude_range, step=step_km*km_to_lat)
    lon_points = np.arange(*longitude_range, step=step_km*km_to_lon)
    lat_points_all, lon_points_all = np.meshgrid(lat_points, lon_points)
    lat_lon = zip(lat_points_all.ravel(), lon_points_all.ravel())
    for lat, lon in lat_lon:
        for page in range(1, max_page):
            # Grid is square (in km) but search is circle
            # Setting step size = search radius will cover entire plane
            json = get_airbnb_json({'page': page, 'latitude': lat, 'longitude': lon,
                                    'maxdistance': step_km})
            data = make_airbnb_json_dataframe(json)
            if data is None:
                print('No more at {lat}, {lon} page {page}'.format(
                                                lat=lat, lon=lon, page=page))
                break  # Go to next location
            print('{len_} listings at {lat}, {lon} page {page}'.format(
                                            len_=len(data), lat=lat, lon=lon,
                                            page=page))
            put_data_in_db(data)


def put_data_in_db(data, table='listings_test',
                   tmp_table='listings_insertion_tmp'):
    """
    Put listing data into MySQL database

    If new and existing listing have same id, listing will be updated with
    new values

    table: table to load into. Must already exist.

    tmp_table: Temporary table for getting around insertion. Must already exist
    with unique constraint issues
    """
    # FIXME
    # See http://stackoverflow.com/questions/24879156/pandas-to-sql-with-sqlalchemy-duplicate-entries-error-in-mysqldb
    # for inspiration of this solution
    engine = db.create_root_engine()
    data.to_sql(tmp_table, engine, if_exists='append', index=False)
    connection = engine.connect()
    connection.execute("REPLACE INTO " + table + " SELECT * FROM " + tmp_table)
    connection.execute('TRUNCATE TABLE ' + tmp_table)

#json = get_airbnb_json({'page': 2, 'latitude': 37.804055, 'longitude': -122.408990,
#                        'maxdistance': 0.1})
#engine = db.create_root_engine()
#for page in range(40, 100):
#   json = get_airbnb_json({'page': page})
#    make_airbnb_json_dataframe(json).to_sql('listingsTest', engine,
#                                           if_exists='append', index=False)
