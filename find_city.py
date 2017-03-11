#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd

shop_info_names=['shop_id','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate_1_name','cate_2_name','cate_3_name']
shop_df = pd.read_csv('shop_info.txt', names=shop_info_names)

w_names=['city_name','day','high_temp','low_temp','detail','wind','wind_level']
w_df = pd.read_csv('city_weather.csv', names=w_names,sep=',')

print w_df.head()



'''
pinyin_names=['name','pinyin']
pinyin_df = pd.read_csv('city_pinyin.txt', names=pinyin_names, sep='    ')
dict = pinyin_df.set_index('name')['pinyin'].to_dict()
shop_df['pinyin'] = shop_df['city_name'].apply(lambda x: dict[x] if (pinyin_df['name'] == x).any() else '###')

print shop_df.head()
'''



