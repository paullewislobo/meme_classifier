import logging
logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/extract_logs.log', level=logging.INFO)
import util
from database import Database
import requests
import uuid
from datetime import datetime
import praw
from dateutil.relativedelta import relativedelta
import time
import argparse

# Setup for PRAW requests. Need to register app with Reddit
client_id = "nO_mDmPsVetqqA"
secret = "El5eOH0mrcTWNq4RbHClcbtsqXo"
user_agent = "linux:com.localhost.memeclassifier:v1.0 (by /u/paullewislobo)"

reddit = praw.Reddit(client_id=client_id,
                     client_secret=secret,
                     user_agent=user_agent)


def process_posts(submissions, subreddit, database, before):
    try:
        data = []
        formatted_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for submission in submissions:
            try:
                if len(submission.url) < 2000:
                    data.append(
                        (
                            str(uuid.uuid4()),
                            formatted_date,
                            formatted_date,
                            "t3_" + submission.id,
                            "https://reddit.com" + submission.permalink.encode('ascii', 'ignore').decode("utf-8"),
                            submission.url.encode('ascii', 'ignore').decode("utf-8"),
                            submission.title.encode('ascii', 'ignore').decode("utf-8"),
                            submission.score,
                            datetime.fromtimestamp(submission.created_utc),
                            str(submission.author_fullname) if submission.author is not None else "",
                            str(submission.author) if submission.author is not None else "",
                            submission.num_comments,
                            "",
                            "",
                            "",
                            "",
                            "",
                            subreddit
                        )
                    )
            except Exception as ex:
                logging.exception("Failed for post" + str(submission.id) + "url" + str(submission.url) +" timestamp " + str(before) + str(ex))
        database.get_cursor().executemany(util.INSERT_MEME_QUERY, data)
    except Exception as ex:
        logging.exception("Error occurred while processing" + str(ex))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subreddit", help="Subreddit to be processed.")
    parser.add_argument("resume", help="Resume processing.", type=bool)
    args = parser.parse_args()
    subreddit = args.subreddit
    resume = args.resume

    database = Database()

    logging.info("Started processing")

    PUSHSHIFT_URL = "https://api.pushshift.io/reddit/submission/search?subreddit=" + subreddit + "&sort_type=created_utc&sort=desc&size=1000"

    start_point = "LAST"

    before = 999999999999999999
    if resume:
        try:
            database.get_cursor().execute("select min(BEFORE_TIMESTAMP) From TRACKER where subreddit=%s",
                                    (subreddit, ))
            temp = database.get_cursor().fetchone()[0]
            if temp is not None:
                before = int(datetime.timestamp(temp))
        except Exception as ex:
            pass
    now = datetime.now()
    years_ago = datetime.now() - relativedelta(years=15)
    stop_point = int(datetime.timestamp(years_ago))
    exception_count = 0
    while before > stop_point:
        try:

            if not before == 999999999999999999:
                url = PUSHSHIFT_URL + "&before=" + str(before)
            else:
                url = PUSHSHIFT_URL
            print(url)
            response = requests.get(url)
            posts = response.json()
            ids_list = set()
            for post in posts['data']:
                ids_list.add('t3_' + post['id'])

            format_strings = ','.join(['%s'] * len(ids_list))

            database.get_cursor().execute(util.CHECK_PROCESSED % format_strings, tuple(ids_list))
            results = database.get_cursor().fetchall()
            results = [row[0] for row in results]
            deduplicated_ids = [id for id in ids_list if id not in results]
            last_post = posts['data'][-1]
            before = last_post['created_utc']
            before = int(float(before))
            submissions = reddit.info(deduplicated_ids)
            process_posts(submissions, subreddit, database, before)
            now = datetime.now()
            formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
            database.get_cursor().execute(util.INSERT_NEXT_CURSOR,
                                        (formatted_date, datetime.fromtimestamp(before), subreddit))

        except Exception as ex:
            logging.exception(ex)
            exception_count += 1
            if exception_count >= 5:
                break
            time.sleep(60)
            continue
