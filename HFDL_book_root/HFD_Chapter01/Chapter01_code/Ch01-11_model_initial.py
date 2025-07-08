# Load the pre-trained BERT model

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
