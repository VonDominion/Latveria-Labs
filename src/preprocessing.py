import re
import string
import nltk

# Download required NLTK resources (safe if already present)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)   # Needed by newer NLTK versions
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize reusable NLP objects
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def remove_html(text):
    """
    Removes HTML tags from a review.

    Example:
    Input : "Great movie! <br /><br /> Loved it."
    Output: "Great movie!  Loved it."
    """
    clean_text = re.sub(r"<.*?>", "", text)
    return clean_text

def remove_urls(text):
    """
    Removes URLs from a review.

    Example:
    Input : "Visit https://example.com for details."
    Output: "Visit  for details."
    """
    clean_text = re.sub(r"http\S+|www\S+", "", text)
    return clean_text

def remove_punctuation(text):
    """
    Removes punctuation marks from a review.

    Example:
    Input : "Amazing movie!!! Must watch."
    Output: "Amazing movie Must watch"
    """
    translator = str.maketrans("","",string.punctuation)
    clean_text = text.translate(translator)
    return clean_text

def remove_numbers(text):
    """
    Removes digits from a review.

    Example:
    Input : "I give this movie 10/10!"
    Output: "I give this movie /!"
    """
    clean_text = re.sub(r"\d+","",text)
    return clean_text

def remove_stopwords(text):
    """
    Removes English stopwords from a review.

    Example:
    Input : "The movie was very good and I loved it"
    Output: "movie good loved"
    """
    words = word_tokenize(text)

    filtered_words = [
        word for word in words
        if word.lower() not in stop_words
    ]

    clean_text = " ".join(filtered_words)

    return clean_text

def lemmatize_text(text):
   """
    Converts words to their base (dictionary) form.

    Example:
    Input : "movies cars children"
    Output: "movie car child"
    """

   words = word_tokenize(text) 

   lemmatized_words = [
       lemmatizer.lemmatize(word)
       for word in words
   ]

   clean_text = " ".join(lemmatized_words)
   return clean_text

def clean_text(text):
    """
    Complete text preprocessing pipeline.

    Steps:
    1. Remove HTML tags
    2. Remove URLs
    3. Convert to lowercase
    4. Remove punctuation
    5. Remove numbers
    6. Remove stopwords
    7. Lemmatize words
    8. Remove extra spaces

    Returns:
        Cleaned review text
    """

    text = remove_html(text)
    text = remove_urls(text)

    # Convert to lowercase
    text = text.lower()

    text = remove_punctuation(text)
    text = remove_numbers(text)

    text = remove_stopwords(text)
    text = lemmatize_text(text)

    # Remove extra spaces
    text = " ".join(text.split())

    return text

