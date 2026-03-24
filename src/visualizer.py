"""
Visualization Module

This module provides functions for creating all visualizations used in the
dashboard including pie charts, bar graphs, word clouds, and statistical plots.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import List, Tuple

# Color scheme for consistent visualization
COLORS = {
    'Positive': '#2ecc71',  # Green
    'Negative': '#e74c3c',  # Red
    'Neutral': '#95a5a6'    # Gray
}


def create_sentiment_pie_chart(df: pd.DataFrame, model: str = 'textblob') -> go.Figure:
    """
    Create a pie chart showing sentiment distribution.

    Args:
        df (pd.DataFrame): Analysis results DataFrame.
        model (str): Which model's results to use ('textblob' or 'vader').
                    Default is 'textblob'.

    Returns:
        go.Figure: Plotly figure object with pie chart.

    Example:
        >>> fig = create_sentiment_pie_chart(results_df, 'vader')
        >>> st.plotly_chart(fig)
    """
    # Get sentiment column based on model
    sentiment_col = f'{model}_sentiment'

    # Count sentiments
    sentiment_counts = df[sentiment_col].value_counts()

    # Ensure all sentiments are present
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment not in sentiment_counts:
            sentiment_counts[sentiment] = 0

    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Negative', 'Neutral'],
        values=[sentiment_counts.get('Positive', 0),
                sentiment_counts.get('Negative', 0),
                sentiment_counts.get('Neutral', 0)],
        marker=dict(colors=[COLORS['Positive'], COLORS['Negative'], COLORS['Neutral']]),
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=f'Sentiment Distribution ({model.capitalize()} Model)',
        title_x=0.5,
        height=400,
        showlegend=True
    )

    return fig


def create_comparison_bar_graph(df: pd.DataFrame) -> go.Figure:
    """
    Create a grouped bar chart comparing TextBlob and VADER results.

    Shows side-by-side comparison of sentiment counts from both models.

    Args:
        df (pd.DataFrame): Analysis results DataFrame.

    Returns:
        go.Figure: Plotly figure object with grouped bar chart.
    """
    # Get sentiment counts for both models
    textblob_counts = df['textblob_sentiment'].value_counts()
    vader_counts = df['vader_sentiment'].value_counts()

    # Ensure all sentiments are present
    sentiments = ['Positive', 'Negative', 'Neutral']
    textblob_values = [textblob_counts.get(s, 0) for s in sentiments]
    vader_values = [vader_counts.get(s, 0) for s in sentiments]

    # Create grouped bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='TextBlob',
        x=sentiments,
        y=textblob_values,
        marker_color=COLORS['Positive'],
        text=textblob_values,
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        name='VADER',
        x=sentiments,
        y=vader_values,
        marker_color=COLORS['Neutral'],
        text=vader_values,
        textposition='auto'
    ))

    fig.update_layout(
        title='Model Comparison: Sentiment Counts',
        title_x=0.5,
        xaxis_title='Sentiment',
        yaxis_title='Count',
        barmode='group',
        height=400,
        showlegend=True
    )

    return fig


def create_score_distribution_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create histogram showing distribution of sentiment scores.

    Shows score distributions for both TextBlob and VADER models.

    Args:
        df (pd.DataFrame): Analysis results DataFrame.

    Returns:
        go.Figure: Plotly figure object with histogram.
    """
    fig = go.Figure()

    # TextBlob scores
    fig.add_trace(go.Histogram(
        x=df['textblob_polarity'],
        name='TextBlob Polarity',
        marker_color=COLORS['Positive'],
        opacity=0.7,
        nbinsx=30
    ))

    # VADER scores
    fig.add_trace(go.Histogram(
        x=df['vader_compound'],
        name='VADER Compound',
        marker_color=COLORS['Negative'],
        opacity=0.7,
        nbinsx=30
    ))

    fig.update_layout(
        title='Sentiment Score Distribution',
        title_x=0.5,
        xaxis_title='Score',
        yaxis_title='Frequency',
        barmode='overlay',
        height=400,
        showlegend=True
    )

    return fig


def create_word_cloud(texts: List[str], sentiment: str = None, width: int = 800, height: int = 400) -> plt.Figure:
    """
    Generate a word cloud from text data.

    Args:
        texts (List[str]): List of texts to generate word cloud from.
        sentiment (str, optional): Filter by sentiment type (not used here,
                                  but kept for API consistency).
        width (int): Width of the word cloud image. Default is 800.
        height (int): Height of the word cloud image. Default is 400.

    Returns:
        plt.Figure: Matplotlib figure containing the word cloud.

    Example:
        >>> fig = create_word_cloud(feedback_texts)
        >>> st.pyplot(fig)
    """
    # Combine all texts
    combined_text = ' '.join([str(text) for text in texts if text])

    if not combined_text or not combined_text.strip():
        # Return empty figure if no text
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.text(0.5, 0.5, 'No text data available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(combined_text)

    # Create figure
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud - Most Frequent Terms', fontsize=14, pad=20)

    return fig


def create_model_agreement_chart(agreement_data: dict) -> go.Figure:
    """
    Create a visualization showing agreement/disagreement between models.

    Args:
        agreement_data (dict): Dictionary from SentimentAnalyzer.compare_models()
                               containing 'agreement_count' and 'disagreement_count'.

    Returns:
        go.Figure: Plotly figure object with agreement chart.
    """
    agreement = agreement_data.get('agreement_count', 0)
    disagreement = agreement_data.get('disagreement_count', 0)

    fig = go.Figure(data=[go.Pie(
        labels=['Agreement', 'Disagreement'],
        values=[agreement, disagreement],
        marker=dict(colors=[COLORS['Positive'], COLORS['Negative']]),
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title='Model Agreement Rate',
        title_x=0.5,
        height=400,
        showlegend=True
    )

    return fig


def create_keyword_frequency_chart(keywords: List[Tuple[str, int]], top_n: int = 10) -> go.Figure:
    """
    Create a horizontal bar chart of most frequent keywords.

    Args:
        keywords (List[Tuple[str, int]]): List of (word, frequency) tuples.
        top_n (int): Number of top keywords to display. Default is 10.

    Returns:
        go.Figure: Plotly figure object with horizontal bar chart.
    """
    # Take top N keywords
    top_keywords = keywords[:top_n]

    if not top_keywords:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title='Top Keywords',
            title_x=0.5,
            xaxis_title='Frequency',
            yaxis_title='Keyword',
            height=400
        )
        return fig

    words = [kw[0] for kw in top_keywords]
    frequencies = [kw[1] for kw in top_keywords]

    # Create horizontal bar chart
    fig = go.Figure(data=[go.Bar(
        x=frequencies,
        y=words,
        orientation='h',
        marker_color=COLORS['Positive'],
        text=frequencies,
        textposition='auto'
    )])

    fig.update_layout(
        title=f'Top {top_n} Most Frequent Keywords',
        title_x=0.5,
        xaxis_title='Frequency',
        yaxis_title='Keyword',
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def create_sentiment_by_category_chart(df: pd.DataFrame, category_col: str = 'category') -> go.Figure:
    """
    Create a stacked bar chart showing sentiment distribution by category.

    Useful when CSV data includes a category column.

    Args:
        df (pd.DataFrame): Analysis results DataFrame with category column.
        category_col (str): Name of the category column. Default is 'category'.

    Returns:
        go.Figure: Plotly figure object with stacked bar chart.
    """
    if category_col not in df.columns:
        # Return empty figure if no category column
        fig = go.Figure()
        fig.update_layout(
            title='Sentiment by Category',
            title_x=0.5,
            height=400
        )
        return fig

    # Group by category and sentiment
    grouped = df.groupby([category_col, 'textblob_sentiment']).size().unstack(fill_value=0)

    # Ensure all sentiment columns exist
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment not in grouped.columns:
            grouped[sentiment] = 0

    # Create stacked bar chart
    fig = go.Figure()

    for sentiment in ['Positive', 'Negative', 'Neutral']:
        fig.add_trace(go.Bar(
            name=sentiment,
            x=grouped.index,
            y=grouped[sentiment],
            marker_color=COLORS[sentiment]
        ))

    fig.update_layout(
        title='Sentiment Distribution by Category',
        title_x=0.5,
        xaxis_title='Category',
        yaxis_title='Count',
        barmode='stack',
        height=400,
        showlegend=True
    )

    return fig


def create_score_comparison_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Create a scatter plot comparing TextBlob and VADER scores.

    Shows how the two models score each text relative to each other.

    Args:
        df (pd.DataFrame): Analysis results DataFrame.

    Returns:
        go.Figure: Plotly figure object with scatter plot.
    """
    fig = go.Figure()

    # Add diagonal reference line (where scores are equal)
    min_score = min(df['textblob_polarity'].min(), df['vader_compound'].min())
    max_score = max(df['textblob_polarity'].max(), df['vader_compound'].max())

    fig.add_trace(go.Scatter(
        x=[min_score, max_score],
        y=[min_score, max_score],
        mode='lines',
        name='Equal Scores',
        line=dict(color='gray', dash='dash'),
        showlegend=True
    ))

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['textblob_polarity'],
        y=df['vader_compound'],
        mode='markers',
        name='Scores',
        marker=dict(
            size=8,
            color=COLORS['Positive'],
            opacity=0.6
        ),
        text=df['text'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x),
        hovertemplate='<b>Text:</b> %{text}<br><br>TextBlob: %{x:.4f}<br>VADER: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title='TextBlob vs VADER Score Comparison',
        title_x=0.5,
        xaxis_title='TextBlob Polarity',
        yaxis_title='VADER Compound Score',
        height=500,
        showlegend=True
    )

    return fig


def create_summary_metrics_cards(summary: dict) -> dict:
    """
    Prepare data for summary metric cards in the dashboard.

    Args:
        summary (dict): Summary dictionary from create_analysis_summary().

    Returns:
        dict: Dictionary with formatted metrics for display cards.
    """
    return {
        'Total Feedback': summary.get('total_feedback', 0),
        'Avg TextBlob Score': f"{summary.get('avg_textblob_score', 0):.4f}",
        'Avg VADER Score': f"{summary.get('avg_vader_score', 0):.4f}",
        'Model Agreement': f"{summary.get('agreement_rate', 0):.1f}%"
    }
