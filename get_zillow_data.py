# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:48:04 2015

@author: ibuder
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

import settings
import db


def zip5(zip_):
    return zip_[0:5]


def make_float(d):
    try:
        return np.float(d)
    except ValueError:
        return np.NaN
    except TypeError:
        return np.NaN


def get_zillow_demographics_one(zip_):
    params = {'zip': zip_, 'zws-id': settings.zws_id}
    page = requests.get(
        'http://www.zillow.com/webservice/GetDemographics.htm',
        params=params)

    root = ET.fromstring(page.text)
    results = {'zip': zip_}
    results['forSale'] = root.findtext('.//forSale')
    results['medianListPricePerSqFt'] = root.findtext(
        ".//*[name='Median List Price Per Sq Ft']/values/zip/value")
    results['homeValueIndex'] = root.findtext(
        ".//*[name='Zillow Home Value Index']/values/zip/value")
    results['medianSingleFamilyHomeValue'] = root.findtext(
        ".//*[name='Median Single Family Home Value']/values/zip/value")
    return results

engine = db.create_root_engine()
# Zillow has API request limit, so try to minimize calls
zipcodes = pd.io.sql.read_sql("""SELECT DISTINCT SUBSTRING(postalCode, 1, 5)
                                     FROM listings WHERE country =
                                     "United States"
                                 """, engine)

zillow_demographics = [get_zillow_demographics_one(zip_)
                       for zip_ in zipcodes.iloc[:, 0]]
zillow_demographics = pd.DataFrame(zillow_demographics)
zillow_demographics.homeValueIndex = zillow_demographics.homeValueIndex.map(
    make_float)
zillow_demographics.medianListPricePerSqFt = \
    zillow_demographics.medianListPricePerSqFt.map(make_float)
zillow_demographics.medianSingleFamilyHomeValue = \
    zillow_demographics.medianSingleFamilyHomeValue.map(make_float)

zillow_demographics.to_sql('zillow_demographics', engine, if_exists='append',
                           index=False)
