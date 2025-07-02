class MovieReviewDataset(Dataset):

    def __init__(self, reviews, sentiments):
        self.reviews = reviews
        self.sentiments = sentiments
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        sentiment = self.sentiments[idx]
        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=512,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {
          'review_text': review,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'labels': torch.tensor(sentiment)
        }

# Prepare the dataset
train_dataset = MovieReviewDataset(train_df['review'].tolist(), train_df['sentiment'].tolist())
test_dataset = MovieReviewDataset(test_df['review'].tolist(), test_df['sentiment'].tolist())
