"""
Utility Functions Module

This module provides helper functions for file handling, data validation,
formatting, and export functionality throughout the application.
"""

import os
import io
from typing import Tuple, List
import pandas as pd
from datetime import datetime


# File paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, 'sample_feedback.csv')


def validate_csv_file(file) -> Tuple[bool, str]:
    """
    Validate an uploaded CSV file.

    Checks if the uploaded file is a valid CSV with at least one text column.
    The CSV should contain a column with feedback text (common names: 'feedback',
    'text', 'comment', 'message', 'student_feedback', etc.)

    Args:
        file: Streamlit uploaded file object.

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
            - is_valid: True if CSV is valid, False otherwise
            - error_message: Empty string if valid, error description if invalid

    Example:
        >>> is_valid, msg = validate_csv_file(uploaded_file)
        >>> if not is_valid:
        ...     st.error(msg)
    """
    if file is None:
        return False, "No file uploaded. Please upload a CSV file."

    # Check file extension
    if not file.name.endswith('.csv'):
        return False, "Invalid file format. Please upload a CSV file."

    # Check file size (warn if larger than 10MB)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    max_size = 10 * 1024 * 1024  # 10MB
    if file_size > max_size:
        return False, f"File is too large ({file_size / 1024 / 1024:.1f}MB). Maximum size is 10MB."

    # Try to read the CSV
    try:
        df = pd.read_csv(file)
        file.seek(0)  # Reset file pointer

        if df.empty:
            return False, "The CSV file is empty. Please upload a file with data."

        # Check for text column
        text_columns = ['feedback', 'text', 'comment', 'message', 'student_feedback',
                       'review', 'response', 'input', 'content']

        found_column = None
        for col in text_columns:
            if col in df.columns:
                found_column = col
                break

        if found_column is None:
            # Provide helpful error message
            available_cols = ', '.join(df.columns.tolist()[:5])
            if len(df.columns) > 5:
                available_cols += ', ...'
            return False, (f"No recognized text column found. Expected columns like: "
                          f"'feedback', 'text', 'comment'. Found: {available_cols}")

        return True, ""

    except pd.errors.EmptyDataError:
        return False, "The CSV file appears to be empty or corrupted."
    except pd.errors.ParserError:
        return False, "Unable to parse the CSV file. Please check the file format."
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def load_sample_data() -> pd.DataFrame:
    """
    Load the sample feedback data from the data directory.

    Returns:
        pd.DataFrame: Sample feedback data with columns:
            - feedback_id
            - student_feedback
            - category
            - timestamp

    Raises:
        FileNotFoundError: If sample data file doesn't exist.
    """
    if os.path.exists(SAMPLE_DATA_PATH):
        return pd.read_csv(SAMPLE_DATA_PATH)
    else:
        raise FileNotFoundError(f"Sample data not found at {SAMPLE_DATA_PATH}")


def get_text_column_from_csv(df: pd.DataFrame) -> str:
    """
    Identify the text column from a CSV DataFrame.

    Searches for common text column names and returns the first match.

    Args:
        df (pd.DataFrame): DataFrame from uploaded CSV.

    Returns:
        str: Name of the text column.

    Raises:
        ValueError: If no text column is found.
    """
    text_columns = ['feedback', 'text', 'comment', 'message', 'student_feedback',
                   'review', 'response', 'input', 'content']

    for col in text_columns:
        if col in df.columns:
            return col

    # Fallback: return first string column
    for col in df.columns:
        if df[col].dtype == 'object':
            return col

    raise ValueError("No text column found in CSV")


def format_score(score: float, precision: int = 4) -> str:
    """
    Format a sentiment score for display.

    Args:
        score (float): The sentiment score to format.
        precision (int): Number of decimal places. Default is 4.

    Returns:
        str: Formatted score as string.

    Example:
        >>> format_score(0.857142857)
        '0.8571'
        >>> format_score(-0.5)
        '-0.5000'
    """
    return f"{score:.{precision}f}"


def get_sentiment_emoji(sentiment: str) -> str:
    """
    Return an emoji representing the sentiment type.

    Args:
        sentiment (str): 'Positive', 'Negative', or 'Neutral'.

    Returns:
        str: Emoji character for the sentiment.

    Example:
        >>> get_sentiment_emoji('Positive')
        '😊'
        >>> get_sentiment_emoji('Negative')
        '😞'
    """
    emoji_map = {
        'Positive': '😊',
        'Negative': '😞',
        'Neutral': '😐'
    }
    return emoji_map.get(sentiment, '')


def get_sentiment_color(sentiment: str) -> str:
    """
    Return a color code representing the sentiment type.

    Uses professional academic colors:
    - Positive: Green (#2ecc71)
    - Negative: Red (#e74c3c)
    - Neutral: Gray (#95a5a6)

    Args:
        sentiment (str): 'Positive', 'Negative', or 'Neutral'.

    Returns:
        str: Hex color code.

    Example:
        >>> get_sentiment_color('Positive')
        '#2ecc71'
    """
    color_map = {
        'Positive': '#2ecc71',
        'Negative': '#e74c3c',
        'Neutral': '#95a5a6'
    }
    return color_map.get(sentiment, '#95a5a6')


def export_results(df: pd.DataFrame, format: str = 'csv') -> bytes:
    """
    Export analysis results for user download.

    Args:
        df (pd.DataFrame): Analysis results to export.
        format (str): Export format - 'csv' or 'excel'. Default is 'csv'.

    Returns:
        bytes: File content as bytes for download.

    Example:
        >>> csv_data = export_results(results_df, 'csv')
        >>> excel_data = export_results(results_df, 'excel')
    """
    if format == 'csv':
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue().encode('utf-8')

    elif format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Analysis Results')
        return output.getvalue()

    else:
        raise ValueError(f"Unsupported export format: {format}")


def create_analysis_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the dashboard.

    Aggregates key metrics for displaying in the dashboard overview.

    Args:
        df (pd.DataFrame): Analysis results DataFrame.

    Returns:
        dict: Summary statistics including:
            - 'total_feedback': Total number of feedback analyzed
            - 'textblob_distribution': Dict with counts of each sentiment
            - 'vader_distribution': Dict with counts of each sentiment
            - 'textblob_percentages': Dict with percentages
            - 'vader_percentages': Dict with percentages
            - 'avg_textblob_score': Average TextBlob polarity
            - 'avg_vader_score': Average VADER compound score
            - 'agreement_rate': Percentage of agreement between models
    """
    total = len(df)

    if total == 0:
        return {
            'total_feedback': 0,
            'textblob_distribution': {},
            'vader_distribution': {},
            'textblob_percentages': {},
            'vader_percentages': {},
            'avg_textblob_score': 0,
            'avg_vader_score': 0,
            'agreement_rate': 0
        }

    # Count distributions
    textblob_dist = df['textblob_sentiment'].value_counts().to_dict()
    vader_dist = df['vader_sentiment'].value_counts().to_dict()

    # Ensure all sentiments are in the dict
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment not in textblob_dist:
            textblob_dist[sentiment] = 0
        if sentiment not in vader_dist:
            vader_dist[sentiment] = 0

    # Calculate percentages
    textblob_pct = {k: round(v / total * 100, 2) for k, v in textblob_dist.items()}
    vader_pct = {k: round(v / total * 100, 2) for k, v in vader_dist.items()}

    # Average scores
    avg_textblob = df['textblob_polarity'].mean()
    avg_vader = df['vader_compound'].mean()

    # Agreement rate
    agreements = (df['textblob_sentiment'] == df['vader_sentiment']).sum()
    agreement_rate = round(agreements / total * 100, 2)

    return {
        'total_feedback': total,
        'textblob_distribution': textblob_dist,
        'vader_distribution': vader_dist,
        'textblob_percentages': textblob_pct,
        'vader_percentages': vader_pct,
        'avg_textblob_score': round(avg_textblob, 4),
        'avg_vader_score': round(avg_vader, 4),
        'agreement_rate': agreement_rate
    }


def get_timestamp() -> str:
    """
    Get current timestamp formatted for display.

    Returns:
        str: Current timestamp in 'YYYY-MM-DD HH:MM:SS' format.
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator (float): The numerator.
        denominator (float): The denominator.
        default (float): Value to return if denominator is zero. Default is 0.0.

    Returns:
        float: Result of division or default value.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with ellipsis suffix.

    Args:
        text (str): Text to truncate.
        max_length (int): Maximum length before truncation. Default is 100.
        suffix (str): Suffix to add if truncated. Default is "...".

    Returns:
        str: Truncated text with suffix if needed.

    Example:
        >>> truncate_text("This is a very long text that needs to be truncated", 30)
        'This is a very long text ...'
    """
    if not text or len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix
