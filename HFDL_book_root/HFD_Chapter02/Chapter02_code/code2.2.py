###### Cleaning text data and handle outliers using Python libraries                 ######
###### focusing on core steps: text preprocessing and filtering based on text length ###### 
###### using the NLTK Library for tokenization and stop-word removal.                ######
###########################################################################################
import re
from nltk.tokenize import word_tokenize
from nltk. corpus import stopwords
# Sample data
text_data = [
 "This is an example!!!",
 "Data cleaning is essential... #NLP",
 "Short",
 "An extraordinarily long sentence that seems to go on forever, which could 
potentially skew the results of an analysis."
]
# Function to clean text data
def clean_text(text):
  # Remove special characters and numbers
  text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
  # Convert to lowercase
  text = text.lower()
  # Tokenize text
  tokens = word_tokenize(text)
  # Remove stopwords
  stop_words = set(stopwords.words('english'))
  tokens = [token for token in tokens if token not in stop_words]
  # Join tokens to recreate the sentence
  return ' '.join(tokens)
# Clean each text in the dataset
cleaned_texts = [clean_text(text) for text in text_data]
# Handling outliers by text length
def handle_outliers(texts, lower_quantile=0.1, upper_quantile=0.9):
 lengths = [len(text.split()) for text in texts]
 lower_bound = sorted(lengths)[int(len(lengths) * lower_quantile)]
 upper_bound = sorted(lengths)[int(len(lengths) * upper_quantile)]
 filtered_texts = [text for text in texts if lower_bound <= len(text.split()) <= 
upper_bound]
 return filtered_texts
# Apply outlier handling
filtered_texts = handle_outliers(cleaned_texts)
print("Cleaned and Filtered Texts:")
for text in filtered_texts:
 print(text)
