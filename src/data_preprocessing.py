import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the "logs" directory exists
log_dir = '../logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column='text', target_column='target'):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        # Define absolute paths using project_root
        raw_data_path = os.path.join(project_root, 'data', 'raw')
        interim_data_path = os.path.join(project_root, 'data', 'interim')
        
        # Load data
        train_data = pd.read_csv(os.path.join(raw_data_path, 'train.csv'))
        test_data = pd.read_csv(os.path.join(raw_data_path, 'test.csv'))
        logger.debug('Raw data loaded')

        # Transform
        train_processed = preprocess_df(train_data, text_column, target_column)
        test_processed = preprocess_df(test_data, text_column, target_column)

        # Save to interim
        os.makedirs(interim_data_path, exist_ok=True)
        train_processed.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', interim_data_path)
        print("Success: Data Preprocessing complete.")

    except Exception as e:
        logger.error('Failed: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()