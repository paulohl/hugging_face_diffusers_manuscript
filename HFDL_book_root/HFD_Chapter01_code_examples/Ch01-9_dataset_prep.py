# Sample dataset

data = {'review': ['I loved the movie!', 'That was the worst movie ever...'],
        'sentiment': [1, 0]}  # 1 for positive, 0 for negative
df = pd.DataFrame(data)

# Splitting the dataset
train_df, test_df = train_test_split(df, test_size=0.25)
