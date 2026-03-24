"""
Text Processing Module

This module provides text preprocessing and cleaning functionality for the
sentiment analysis pipeline. It includes functions for:
- Text cleaning and normalization
- Keyword extraction
- Text preparation for word cloud generation
"""

import re
from typing import List, Tuple
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Download required NLTK data (only if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Clean and normalize text for sentiment analysis.

    This function performs basic text preprocessing including:
    - Removing extra whitespace
    - Removing URLs and email addresses
    - Removing special characters (optional, keeping basic punctuation)
    - Converting to lowercase (optional)

    Args:
        text (str): The text to clean.
        lowercase (bool): Whether to convert text to lowercase. Default is True.

    Returns:
        str: Cleaned and normalized text.

    Example:
        >>> clean_text("  Check out https://example.com!!!  ")
        'check out'
    """
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Optional: Convert to lowercase
    if lowercase:
        text = text.lower()

    return text.strip()


def extract_keywords(texts: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Extract the most frequent keywords from a list of texts.

    This function tokenizes all texts, removes stop words, and counts
    word frequencies to identify the most common keywords.

    Args:
        texts (List[str]): List of texts to analyze.
        top_n (int): Number of top keywords to return. Default is 10.

    Returns:
        List[Tuple[str, int]]: List of (word, frequency) tuples sorted by frequency.

    Example:
        >>> texts = ["Great course content", "Great instructor", "Good course"]
        >>> extract_keywords(texts, top_n=3)
        [('great', 2), ('course', 2), ('content', 1)]
    """
    # Get stop words
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback if stopwords not available
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                      'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                      'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for',
                      'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                      'before', 'after', 'above', 'below', 'between', 'under', 'again',
                      'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                      'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                      'too', 'very', 'just', 'this', 'that', 'these', 'those', 'am', 'it',
                      'its', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                      'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                      'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their',
                      'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'about'}

    # Combine all texts and tokenize
    all_words = []
    for text in texts:
        if text and text.strip():
            # Clean and tokenize
            clean = clean_text(text, lowercase=True)
            try:
                tokens = word_tokenize(clean)
                # Filter out stop words and short words, keep only alphabetic
                words = [word for word in tokens
                        if word not in stop_words
                        and len(word) > 2
                        and word.isalpha()]
                all_words.extend(words)
            except:
                # Fallback: simple split if tokenization fails
                words = [word for word in clean.split()
                        if word not in stop_words
                        and len(word) > 2
                        and word.isalpha()]
                all_words.extend(words)

    # Count frequencies
    word_counts = Counter(all_words)

    # Return top N most common words
    return word_counts.most_common(top_n)


def prepare_text_for_wordcloud(texts: List[str], sentiment_filter: str = None) -> str:
    """
    Prepare and combine texts for word cloud generation.

    This function cleans texts, optionally filters by sentiment, and combines
    them into a single string suitable for word cloud generation.

    Args:
        texts (List[str]): List of texts to prepare.
        sentiment_filter (str, optional): Not used in this function but kept
                                         for API consistency. Can be used to
                                         filter texts by sentiment before calling.

    Returns:
        str: Combined text string ready for word cloud generation.

    Example:
        >>> texts = ["Great course!", "Excellent content"]
        >>> prepare_text_for_wordcloud(texts)
        'great course excellent content'
    """
    # Clean all texts
    cleaned_texts = []
    for text in texts:
        if text and text.strip():
            # Keep some punctuation for word cloud context
            clean = clean_text(text, lowercase=True)
            cleaned_texts.append(clean)

    # Combine all texts
    combined_text = ' '.join(cleaned_texts)

    return combined_text


def get_text_statistics(text: str) -> dict:
    """
    Get basic statistics about a text.

    Provides information such as word count, character count,
    and sentence count.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: Dictionary containing:
            - 'word_count' (int): Number of words
            - 'char_count' (int): Number of characters (including spaces)
            - 'char_count_no_spaces' (int): Number of characters (excluding spaces)
            - 'sentence_count' (int): Number of sentences
            - 'avg_word_length' (float): Average word length
    """
    if not text or not text.strip():
        return {
            'word_count': 0,
            'char_count': 0,
            'char_count_no_spaces': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0
        }

    # Basic counts
    word_count = len(text.split())
    char_count = len(text)
    char_count_no_spaces = len(text.replace(' ', ''))

    # Sentence count using TextBlob
    try:
        blob = TextBlob(text)
        sentence_count = len(blob.sentences)
    except:
        # Fallback: count by periods, exclamation marks, question marks
        sentence_count = len(re.findall(r'[.!?]+', text))
        if sentence_count == 0 and word_count > 0:
            sentence_count = 1

    # Average word length
    if word_count > 0:
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / word_count
    else:
        avg_word_length = 0.0

    return {
        'word_count': word_count,
        'char_count': char_count,
        'char_count_no_spaces': char_count_no_spaces,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 2)
    }


def validate_text_input(text: str, min_length: int = 3, max_length: int = 5000) -> Tuple[bool, str]:
    """
    Validate text input before processing.

    Checks if text meets minimum and maximum length requirements
    and is not empty or just whitespace.

    Args:
        text (str): The text to validate.
        min_length (int): Minimum number of characters. Default is 3.
        max_length (int): Maximum number of characters. Default is 5000.

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
            - is_valid: True if text is valid, False otherwise
            - error_message: Empty string if valid, error description if invalid

    Example:
        >>> validate_text_input("Hi")
        (False, 'Text is too short. Please enter at least 3 characters.')
        >>> validate_text_input("This is a valid text.")
        (True, '')
    """
    if not text or not text.strip():
        return False, "Text cannot be empty. Please enter some text to analyze."

    text_length = len(text.strip())

    if text_length < min_length:
        return False, f"Text is too short. Please enter at least {min_length} characters."

    if text_length > max_length:
        return False, f"Text is too long. Maximum {max_length} characters allowed."

    return True, ""
