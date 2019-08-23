""" ------------------- Queries -----------------------"""

INSERT_TEMPLATE_QUERY = "INSERT INTO TEMPLATE(FILE_PATH, TITLE, DOWNLOAD_URL) VALUES(%s, %s, %s, %s)"

INSERT_MEME_QUERY = """INSERT INTO MEME
    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

UPDATE_MEME_QUERY = """UPDATE MEME 
                        SET PREDICTED_PERCENTAGE = %s,
                        PREDICTED_CLASS = %s,
                        PREDICTED_SCORE = %s,
                        CLASS_MEAN_SCORE = %s,
                        PREDICTION = %s
                        WHERE MEME_ID = %s"""

UPDATE_DELETED_MEME_QUERY = """UPDATE MEME 
                        SET PREDICTION = 'REMOVED'
                        WHERE MEME_ID = %s"""


INSERT_TAG_QUERY = "INSERT INTO TAG VALUES(%s, %s, %s, %s)"

INSERT_NEXT_CURSOR = "INSERT INTO TRACKER VALUES(%s, %s, %s)"

GET_LAST_TIMESTAMP = """
    select min(BEFORE_TIMESTAMP) From TRACKER
"""

GET_START_CURSOR = """
    select * From (
        select * 
            From TRACKER
            where subreddit = %s 
            order by iteration asc, id desc
        ) AS track
    limit 1
"""

GET_TEMPLATE_QUERY = """SELECT T.TITLE, T.FILE_PATH, COUNT(M.MEME_ID) AS COUNT FROM TEMPLATE T LEFT OUTER JOIN MEME M ON M.PREDICTION = T.TITLE GROUP BY T.TITLE, T.FILE_PATH order by COUNT(M.MEME_ID) desc, T.TITLE ASC"""

GET_TEMPLATES = """SELECT T.ID, T.TITLE FROM TEMPLATE T"""

GET_MEMES_FOR_TEMPLATE = "SELECT ID, TITLE, URL, IMAGE_URL, PREDICTED_PERCENTAGE, PREDICTED_CLASS, PREDICTED_SCORE, CLASS_MEAN_SCORE, PREDICTION FROM MEME WHERE TRIM(PREDICTION) = TRIM(%(template_name)s) and PREDICTION <> '' and PREDICTED_PERCENTAGE > %(score)s LIMIT %(lower_limit)s, %(upper_limit)s"

CHECK_PROCESSED = """SELECT distinct ID FROM MEME WHERE ID IN (%s)"""

GET_MEMES_FOR_PROCESSING = """
SELECT * FROM MEME WHERE PREDICTION = "" ORDER BY CREATION_TS DESC LIMIT 4000;
"""


""" ------------------- Constants -----------------------"""

CURRENT = "CURRENT"
LAST = "LAST"
TEMPLATE_PATH = "meme_templates/"
IMG_FLIP_BASE_URL = "https://imgflip.com"
PAGE = "?page="
SEARCH = "/search?q="



