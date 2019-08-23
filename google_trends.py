import requests
from bs4 import BeautifulSoup
import time
import praw
import datetime
from time import mktime
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
import pandas as pd
import random

from database import Database
import time

import glob
import os

files = glob.glob("trends/*")
completed = set()
for file in files:
    file_name = os.path.splitext(file)[0]
    file_name = os.path.basename(file_name)
    completed.add(file_name)
print(completed)

database = Database()
database_connection = database.get_connection()
templates = pd.read_sql("SELECT * FROM TEMPLATE ", con=database_connection)

kw_list = []
for index, template in templates.iterrows():
    try:
        wait = False
        print(str(template['ID']))
        if str(template['ID']) not in completed:
            print("processing", str(template['ID']))
            title = template['TITLE'][:-5]
            kw_list = [title]
            pytrend = TrendReq()
            pytrend.build_payload(kw_list)
            interest_over_time_df = pytrend.interest_over_time()
            ax = plt.gca()
            interest_over_time_df.plot(kind='line', y=title, ax=ax)
            plt.draw()
            plt.savefig('trends/' + str(template['ID']) + '.png', dpi=300)
            interest_over_time_df.to_csv('trends/' + str(template['ID']) + '.csv')
            plt.close()
            wait = True
    except Exception as ex:
        print("Failed for ", str(template['TITLE']), ex)
        wait = True
    finally:
        if wait:
            time.sleep(random.choice(range(25, 30)))