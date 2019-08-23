import logging
logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/logs.log', level=logging.ERROR)
from tensorflow.python.keras.models import Model
import util
from classifier import Classifier
import urllib
import urllib.request
import database
import numpy as np
import pandas as pd
import cv2
import h5py
from database import Database
from multiprocessing import Pool
import time
import os


def process(model, records):
    image_ref = dict()
    images = []
    i = 0
    for record in records:
        try:
            meme_id = record[0]
            image_url = record[5]

            resp = urllib.request.urlopen(image_url)
            img = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(300, 300))
            images.append(img)
            image_ref[meme_id] = i
            i += 1
        except Exception as ex:
            logging.exception("Failed to process ", str(meme_id))
        images = np.array(images)
        images/255

        predict = classifier.model.predict(images)
        intermediate_score = intermediate_layer_model.predict(images)

        data = []
        for k, v in image_ref:
            index = np.argmax(predict[v])
            percentage = np.max(predict[v])
            pre_softmax_score = intermediate_score[v][index]

            predicted_percentage = str(percentage)  # PREDICTED_PERCENTAGE
            predicted_class = template[template['ID'] == index + 1]['TITLE'].to_string(index=False)  # PREDICTED_CLASS
            pre_softmax_score = str(pre_softmax_score)
            class_mean_score = average_scores[index]
            if percentage > 0.98 and (pre_softmax_score > average_scores[index] * 0.95) and (pre_softmax_score < average_scores[index] * 1.05):
                prediction = template[template['ID'] == index + 1]['TITLE'].to_string(index=False)
            else:
                prediction = 'Unknown'
            data.append((
                predicted_percentage,
                predicted_class,
                pre_softmax_score,
                class_mean_score,
                prediction,
                meme_id
            ))

        database.get_cursor().executemany(util.UPDATE_MEME_QUERY, data)


def divide_into_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def retrieve_images(record):
    img = None
    try:
        meme_id = record[0]
        image_url = record[5]
        logging.info("Processing id" + str( image_url))

        if image_url.endswith(".png") or image_url.endswith(".jpg"):
            resp = urllib.request.urlopen(image_url, timeout=10)
            img = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(300, 300))
        logging.info("Completed id" +str(image_url))
    except Exception as ex:
        img = None
        logging.info("Exception occurred for" + str(image_url))

    return img


if __name__ == "__main__":
    logging.info("Started processing")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    PROCESS = True

    database = Database()
    try:
        start_time = time.time()
        templates = pd.read_sql(util.GET_TEMPLATES, con=database.get_connection())
        # Create Model
        classifier = Classifier()
        classifier.load_model()

        current_time = time.time()
        print("Loaded Model", current_time - start_time)
        start_time = current_time

        hf = h5py.File('x_train_1.h5', 'r')
        x_train_1 = hf.get('x_train_1')
        x_train_1 = np.array(x_train_1, dtype=np.float32) / 255

        hf = h5py.File('y_train_1.h5', 'r')
        y_train_1 = hf.get('y_train_1')
        y_train_1 = np.array(y_train_1)

        hf = h5py.File('x_train_2.h5', 'r')
        x_train_2 = hf.get('x_train_2')
        x_train_2 = np.array(x_train_2, dtype=np.float32) / 255

        hf = h5py.File('y_train_2.h5', 'r')
        y_train_2 = hf.get('y_train_2')
        y_train_2 = np.array(y_train_2)

        hf = h5py.File('x_test.h5', 'r')
        x_test = hf.get('x_test')
        x_test = np.array(x_test, dtype=np.float32) / 255

        hf = h5py.File('y_test.h5', 'r')
        y_test = hf.get('y_test')
        y_test = np.array(y_test)

        current_time = time.time()
        print("Loaded Files", current_time - start_time)
        start_time = current_time

        # if TRAIN:
        #     logging.info("Started Training")
        #
        #     for i in range(100):
        #         print("EPOCH", i)
        #         classifier.train(x_train_1, y_train_1, epochs=1, batch_size=100)
        #         classifier.train(x_train_2, y_train_2, epochs=1, batch_size=100)
        #
        #         if i % 10 == 0:
        #             print("Evaluation")
        #             classifier.evaluate(x_test, y_test)

        layer_name = 'my_layer'
        intermediate_layer_model = Model(inputs=classifier.model.input,
                                         outputs=classifier.model.layers[-2].output)
        intermediate_1 = intermediate_layer_model.predict(x_train_1)
        intermediate_2 = intermediate_layer_model.predict(x_train_2)
        predicted_1 = classifier.model.predict(x_train_1)
        predicted_2 = classifier.model.predict(x_train_2)

        one_hot_1 = (np.eye(825)[y_train_1]).reshape(-1, 825)
        one_hot_2 = (np.eye(825)[y_train_2]).reshape(-1, 825)
        target_1 = intermediate_1 * one_hot_1
        target_2 = intermediate_2 * one_hot_2

        sum = np.sum(target_1, axis=0) + np.sum(target_2, axis=0)
        count = (np.count_nonzero(target_1, axis=0) + (np.count_nonzero(target_2, axis=0) + 0.000001))

        average_scores = sum / count

        del x_test
        del x_train_1
        del x_train_2

        current_time = time.time()
        print("Calculated Scores", current_time - start_time)
        start_time = current_time

        if PROCESS:
            logging.info("Started processing")
            start_time = time.time()
            database = Database()
            while True:
                database.get_cursor().execute(util.GET_MEMES_FOR_PROCESSING)
                results = database.get_cursor().fetchall()

                if len(results) == 0:
                    break

                current_time = time.time()
                print("Database Query", current_time - start_time)
                start_time = current_time

                with Pool(16) as p:
                    images = p.map(retrieve_images, results)


                #images = []
                #for image in results:
                #    images.append(retrieve_images(image))

                current_time = time.time()
                print("Parallel Get Requests", current_time - start_time)
                start_time = current_time

                final_images = []
                i = 0
                image_ref = dict()

                failed_posts = []

                for j in range(len(results)):
                    if images[j] is not None:
                        image_ref[results[j][0]] = i
                        final_images.append(images[j])
                        i += 1
                    else:
                        failed_posts.append((results[j][0],))
                print("Total images to be processed", len(final_images))
                current_time = time.time()
                print("For Loop", current_time - start_time)
                start_time = current_time
                final_images = np.array(final_images)
                final_images = final_images / 255

                current_time = time.time()
                print("Divide by 255", current_time - start_time)
                start_time = current_time

                predict = classifier.model.predict(final_images)
                intermediate_score = intermediate_layer_model.predict(final_images)

                data = []
                for k, v in image_ref.items():
                    index = np.argmax(predict[v])
                    percentage = np.max(predict[v])
                    pre_softmax_score = intermediate_score[v][index]

                    predicted_percentage = str(percentage)  # PREDICTED_PERCENTAGE
                    predicted_class = templates[templates['ID'] == index + 1]['TITLE'].to_string(
                        index=False)  # PREDICTED_CLASS
                    pre_softmax_score = pre_softmax_score
                    class_mean_score = average_scores[index]
                    if percentage > 0.98 and (pre_softmax_score > average_scores[index] * 0.95) and (
                            pre_softmax_score < average_scores[index] * 1.05):
                        prediction = templates[templates['ID'] == index + 1]['TITLE'].to_string(index=False)
                    else:
                        prediction = 'Unknown'

                    data.append(
                        (
                            str(predicted_percentage),
                            str(predicted_class),
                            str(pre_softmax_score),
                            str(class_mean_score),
                            str(prediction),
                            k
                        )
                    )

                current_time = time.time()
                print("Model", current_time - start_time)
                start_time = current_time
                if len(data) > 0:
                    database.get_cursor().executemany(util.UPDATE_MEME_QUERY, data)
                if len(failed_posts) > 0:
                    database.get_cursor().executemany(util.UPDATE_DELETED_MEME_QUERY, failed_posts)

    except Exception as ex:
        logging.exception("Exception occurred")
    finally:
        del database
