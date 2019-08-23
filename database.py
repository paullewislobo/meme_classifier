import mysql.connector

# def get_connection():
#     database_username = 'MEME'
#     database_password = 'MEME'
#     database_ip       = 'localhost:3306'
#     database_name     = 'MEME'
#     database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
#                                                    format(database_username, database_password,
#                                                           database_ip, database_name))
#     return database_connection


class Database:
    def __init__(self):
        self.conn = mysql.connector.connect(user='MEME',
                                            password='MEME',
                                            database='MEME',
                                            host='localhost',
                                            port='3306',
                                            autocommit=True)
        self.cursor = self.conn.cursor()

    def get_cursor(self):
        return self.cursor

    def get_connection(self):
        return self.conn

    def __del__(self):
        self.cursor.close()
        self.conn.close()
