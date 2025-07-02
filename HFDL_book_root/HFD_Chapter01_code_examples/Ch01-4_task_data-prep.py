import pandas as pd
from sklearn.model_selection import train_test_split

data = {
    'review': ['I loved the movie!', 'That was the worst movie ever...'],
    'sentiment': [1, 0]  # 1 = positive, 0 = negative
}

df = pd.DataFrame(data)
train_df, test_df = train_test_split(df, test_size=0.25)