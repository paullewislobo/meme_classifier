# imports
from bs4 import BeautifulSoup
import json
import requests
import os
import urllib
from urllib.parse import urlparse
import urllib.request
import database
import uuid
from datetime import datetime
import sqlalchemy
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from skimage import measure
import h5py
import util

database_connection = database.get_connection()

IMG_FLIP_URL = "https://imgflip.com"

MEME_TEMPLATE_URL = "/memetemplates?page="

INSERT_TEMPLATE_QUERY = "INSERT INTO TEMPLATE(FILE_PATH, TITLE, DOWNLOAD_URL) VALUES(%s, %s, %s)"

PAGE_URL = "?page="

HTTPS = "https:"

TRAINING_PATH = "static/training/"
TESTING_PATH = "static/testing/"

# Loop through pages from imgflip
all_templates = set()
for i in range(1, 22):
    response = requests.get(IMG_FLIP_URL + MEME_TEMPLATE_URL + str(i))
    soup = BeautifulSoup(response.content, 'html.parser')

    # Get all templates for list
    template_list = soup.findAll("div", {"class": "mt-img-wrap"})

    for link in template_list:
        a = link.find("a")
        meme_url = IMG_FLIP_URL + a["href"]

        title = a["title"]
        if title not in all_templates:
            all_templates.add(title)
            response = requests.get(meme_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            right_panel = soup.find("div", {"id": "base-right"})
            templates = right_panel.select_one("img[alt*=Blank]")
            template_url = IMG_FLIP_URL + templates["src"]

            a = urlparse(template_url)
            file_name = os.path.basename(a.path)
            # INSERT TEMPLATE INTO DATABASE

            path = 'static/meme_templates/' + file_name
            with open(path, 'wb') as f:
                f.write(requests.get(template_url).content)
            database_connection.execute(INSERT_TEMPLATE_QUERY, ('meme_templates/' + file_name, title, template_url))

            if not os.path.exists(TRAINING_PATH + title):
                os.makedirs(TRAINING_PATH + title)
            else:
                continue

            if not os.path.exists(TESTING_PATH + title):
                os.makedirs(TESTING_PATH + title)
            else:
                continue

            j = 1
            for k in range(1, 6):
                try:
                    data_url = meme_url + PAGE_URL + str(k)
                    response = requests.get(data_url)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    left_panel = soup.find("div", {"id": "base-left"})
                    images = left_panel.findAll("img")
                    for image in images:
                        image_url = HTTPS + image["src"]

                        a = urlparse(image_url)
                        file_name = os.path.basename(a.path)

                        if j % 5 == 0:
                            path = TESTING_PATH + title + "/" + file_name
                        else:
                            path = TRAINING_PATH + title + "/" + file_name

                        with open(path, 'wb') as f:
                            f.write(requests.get(image_url).content)
                        j += 1
                except Exception as ex:
                    print("Error occurred while retrieving data for", meme_url + PAGE_URL + str(k))
