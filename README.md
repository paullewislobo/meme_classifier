# Environment Setup

1. Setup MySql
- Refer to the following link:
```https://dev.mysql.com/doc/refman/8.0/en/installing.html```

- Setup MySQl default password
```http://www.ihp.sinica.edu.tw/dashboard/docs/reset-mysql-password.html```

- Setup user MEME with password MEME
```GRANT ALL PRIVILEGES ON *.* TO 'MEME'@'localhost' IDENTIFIED BY 'MEME';```

- Update database.py file with credentials if changed.

2. Install Anaconda
```https://docs.anaconda.com/anaconda/install/```

3. Create Virtual Environment from environment file
```conda env create -f environment.yml```

4. Activate Environment
- Linux
```source activate meme```
- Windows
```activate meme```

4. Setup Github
- Refer to the following link
``` https://help.github.com/en/articles/set-up-git```

5. Clone git repository

- Clone
```git clone TODO```
- Change directory
```TODO```

6. Setup Praw API

- Create Reddit developer account
```https://praw.readthedocs.io/en/latest/getting_started/authentication.html```
- Update client_id, client_secret and user_agent into extract_data.py

# Populate templates and training data
```python populate_dataset.py```

# Train Classifier
```python classifier.py```

# Extract Data from Push shift
```python extract.py```

# Process extracted data
```python processor.py```

# Visualize the processed memes
```python reports.py```