# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:05:44 2015

@author: ibuder
"""

import sqlalchemy


def create_root_engine():
    return sqlalchemy.create_engine(
        'mysql+pymysql://root:' +
        '@localhost/airbnb?charset=utf8')
