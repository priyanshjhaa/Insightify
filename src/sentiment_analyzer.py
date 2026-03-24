"""
Sentiment Analysis Module

This module provides sentiment analysis functionality using two different NLP models:
1. TextBlob: A lexicon-based approach that returns polarity and subjectivity scores
2. VADER: A rule-based model optimized for social media and short text

The module implements the SentimentAnalyzer class which provides:
- Single text analysis
- Batch processing
- Model comparison and agreement analysis
"""

from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """
    Main class for performing sentiment analysis using TextBlob and VADER models.

    This class provides methods to analyze individual texts or batches of texts,
    compare results from both models, and generate comparative statistics.

    Attributes:
        textblob_analyzer: TextBlob analyzer (used implicitly via TextBlob class)
        vader_analyzer: VADER sentiment intensity analyzer instance
    """

    def __init__(self):
        """
        Initialize the SentimentAnalyzer with both VADER and TextBlob models.

        VADER is initialized explicitly while TextBlob is used via its main class.
        Both models are ready for analysis upon initialization.
        """
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def analyze_single_text(self, text: str) -> Dict:
        """
        Analyze a single text input using both TextBlob and VADER models.

        This method processes the text through both sentiment analysis models
        and returns comprehensive results including scores and classifications.

        Args:
            text (str): The text to analyze. Should be a non-empty string.

        Returns:
            dict: A dictionary containing:
                - 'text' (str): The original text
                - 'textblob' (dict): TextBlob results with:
                    - 'polarity' (float): Score from -1 (negative) to +1 (positive)
                    - 'subjectivity' (float): Score from 0 (objective) to 1 (subjective)
                    - 'sentiment' (str): 'Positive', 'Negative', or 'Neutral'
                - 'vader' (dict): VADER results with:
                    - 'compound' (float): Normalized compound score (-1 to +1)
                    - 'neg' (float): Negative sentiment proportion
                    - 'neu' (float): Neutral sentiment proportion
                    - 'pos' (float): Positive sentiment proportion
                    - 'sentiment' (str): 'Positive', 'Negative', or 'Neutral'
                - 'timestamp' (datetime): When the analysis was performed

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> result = analyzer.analyze_single_text("I love this course!")
            >>> print(result['textblob']['sentiment'])
            'Positive'
        """
        # Clean the text (basic cleaning)
        clean_text = text.strip()

        # TextBlob Analysis
        # TextBlob uses a lexicon-based approach for sentiment analysis
        blob = TextBlob(clean_text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        textblob_sentiment = self._classify_sentiment(textblob_polarity)

        # VADER Analysis
        # VADER is specifically tuned for social media and short texts
        vader_scores = self.vader_analyzer.polarity_scores(clean_text)
        vader_compound = vader_scores['compound']
        vader_sentiment = self._classify_sentiment(vader_compound)

        return {
            'text': clean_text,
            'textblob': {
                'polarity': textblob_polarity,
                'subjectivity': textblob_subjectivity,
                'sentiment': textblob_sentiment
            },
            'vader': {
                'compound': vader_compound,
                'neg': vader_scores['neg'],
                'neu': vader_scores['neu'],
                'pos': vader_scores['pos'],
                'sentiment': vader_sentiment
            },
            'timestamp': datetime.now()
        }

    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze multiple texts efficiently and return results as a DataFrame.

        This method processes a list of texts through both models and returns
        a structured DataFrame with all results for easy analysis and visualization.

        Args:
            texts (List[str]): List of texts to analyze.

        Returns:
            pd.DataFrame: A DataFrame with columns:
                - 'text': Original text
                - 'textblob_polarity', 'textblob_subjectivity', 'textblob_sentiment'
                - 'vader_compound', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_sentiment'
                - 'timestamp': Analysis timestamp

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> texts = ["Great course!", "Not good", "It's okay"]
            >>> df = analyzer.analyze_batch(texts)
            >>> print(df[['text', 'textblob_sentiment', 'vader_sentiment']])
        """
        results = []

        for text in texts:
            if text and text.strip():  # Skip empty texts
                result = self.analyze_single_text(text)
                results.append({
                    'text': result['text'],
                    'textblob_polarity': result['textblob']['polarity'],
                    'textblob_subjectivity': result['textblob']['subjectivity'],
                    'textblob_sentiment': result['textblob']['sentiment'],
                    'vader_compound': result['vader']['compound'],
                    'vader_neg': result['vader']['neg'],
                    'vader_neu': result['vader']['neu'],
                    'vader_pos': result['vader']['pos'],
                    'vader_sentiment': result['vader']['sentiment'],
                    'timestamp': result['timestamp']
                })

        return pd.DataFrame(results)

    def _classify_sentiment(self, score: float) -> str:
        """
        Classify sentiment based on polarity score threshold.

        Classification rules:
        - score > 0: Positive
        - score < 0: Negative
        - score = 0: Neutral

        Args:
            score (float): Polarity or compound score from sentiment model.

        Returns:
            str: 'Positive', 'Negative', or 'Neutral'
        """
        if score > 0:
            return 'Positive'
        elif score < 0:
            return 'Negative'
        else:
            return 'Neutral'

    def compare_models(self, df: pd.DataFrame) -> Dict:
        """
        Generate comparative statistics between TextBlob and VADER models.

        This method analyzes how often the two models agree or disagree and
        provides insights into their comparative behavior.

        Args:
            df (pd.DataFrame): DataFrame output from analyze_batch() method.
                               Must contain 'textblob_sentiment' and 'vader_sentiment' columns.

        Returns:
            dict: A dictionary containing:
                - 'agreement_count' (int): Number of texts where models agree
                - 'disagreement_count' (int): Number of texts where models disagree
                - 'agreement_percentage' (float): Percentage of agreement
                - 'textblob_distribution' (dict): Sentiment distribution from TextBlob
                - 'vader_distribution' (dict): Sentiment distribution from VADER
                - 'average_scores' (dict): Mean scores from both models
                - 'disagreement_cases' (pd.DataFrame): Cases where models disagreed

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> df = analyzer.analyze_batch(texts)
            >>> comparison = analyzer.compare_models(df)
            >>> print(f"Models agree {comparison['agreement_percentage']:.1f}% of the time")
        """
        # Count agreements and disagreements
        agreements = (df['textblob_sentiment'] == df['vader_sentiment']).sum()
        disagreements = (df['textblob_sentiment'] != df['vader_sentiment']).sum()

        # Calculate percentage
        total = len(df)
        agreement_percentage = (agreements / total * 100) if total > 0 else 0

        # Get sentiment distributions
        textblob_dist = df['textblob_sentiment'].value_counts().to_dict()
        vader_dist = df['vader_sentiment'].value_counts().to_dict()

        # Calculate average scores
        avg_scores = {
            'textblob_avg_polarity': df['textblob_polarity'].mean(),
            'vader_avg_compound': df['vader_compound'].mean()
        }

        # Get disagreement cases
        disagreement_cases = df[df['textblob_sentiment'] != df['vader_sentiment']].copy()

        return {
            'agreement_count': int(agreements),
            'disagreement_count': int(disagreements),
            'agreement_percentage': round(agreement_percentage, 2),
            'textblob_distribution': textblob_dist,
            'vader_distribution': vader_dist,
            'average_scores': avg_scores,
            'disagreement_cases': disagreement_cases
        }

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive summary statistics from analyzed data.

        Provides key metrics for dashboard display including total count,
        sentiment percentages, and model comparison metrics.

        Args:
            df (pd.DataFrame): DataFrame output from analyze_batch() method.

        Returns:
            dict: Summary statistics including:
                - 'total_analyzed' (int): Total number of texts processed
                - 'textblob_percentages' (dict): Sentiment percentages from TextBlob
                - 'vader_percentages' (dict): Sentiment percentages from VADER
                - 'average_subjectivity' (float): Average subjectivity score
                - 'model_comparison' (dict): Results from compare_models()
        """
        total = len(df)

        # Calculate percentages for each model
        textblob_pct = {}
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            count = (df['textblob_sentiment'] == sentiment).sum()
            textblob_pct[sentiment] = round(count / total * 100, 2) if total > 0 else 0

        vader_pct = {}
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            count = (df['vader_sentiment'] == sentiment).sum()
            vader_pct[sentiment] = round(count / total * 100, 2) if total > 0 else 0

        return {
            'total_analyzed': total,
            'textblob_percentages': textblob_pct,
            'vader_percentages': vader_pct,
            'average_subjectivity': round(df['textblob_subjectivity'].mean(), 3),
            'model_comparison': self.compare_models(df)
        }
