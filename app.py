"""
Student Feedback Sentiment Analysis Dashboard

A comprehensive NLP application that analyzes student feedback using
TextBlob and VADER sentiment analysis models with comparative visualizations.

Author: NLP Project Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.sentiment_analyzer import SentimentAnalyzer
from src.text_processor import clean_text, extract_keywords, prepare_text_for_wordcloud, validate_text_input
from src.utils import (
    validate_csv_file, load_sample_data, get_text_column_from_csv,
    format_score, get_sentiment_emoji, get_sentiment_color,
    export_results, create_analysis_summary, get_timestamp
)
from src.visualizer import (
    create_sentiment_pie_chart, create_comparison_bar_graph,
    create_score_distribution_plot, create_word_cloud,
    create_model_agreement_chart, create_keyword_frequency_chart,
    create_score_comparison_scatter, create_summary_metrics_cards
)


# Page configuration
st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic presentation
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        color: #2ecc71;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #e74c3c;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #95a5a6;
        font-weight: bold;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """
    Initialize session state variables for data persistence across tabs.
    """
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SentimentAnalyzer()

    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    if 'current_texts' not in st.session_state:
        st.session_state.current_texts = []


def render_header():
    """
    Render the main header and description of the application.
    """
    st.markdown('<div class="main-header">🎓 Student Feedback Sentiment Analysis Dashboard</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    This dashboard analyzes student feedback using two NLP models:
    <b>TextBlob</b> (lexicon-based) and <b>VADER</b> (optimized for social text).
    Compare results, visualize distributions, and extract insights from feedback data.
    </div>
    """, unsafe_allow_html=True)


def tab_single_analysis():
    """
    Tab 1: Single text analysis with real-time results.
    """
    st.markdown('<div class="sub-header">📝 Single Text Analysis</div>', unsafe_allow_html=True)

    # Text input area
    text_input = st.text_area(
        "Enter student feedback to analyze:",
        placeholder="Type or paste the feedback text here...",
        height=150,
        max_chars=5000,
        help="Enter any text between 3 and 5000 characters"
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.session_state.single_result = None
        st.rerun()

    if analyze_btn:
        # Validate input
        is_valid, error_msg = validate_text_input(text_input)

        if not is_valid:
            st.error(f"❌ {error_msg}")
            return

        # Analyze
        with st.spinner("Analyzing sentiment..."):
            result = st.session_state.analyzer.analyze_single_text(text_input)
            st.session_state.single_result = result

    # Display results
    if 'single_result' in st.session_state and st.session_state.single_result:
        result = st.session_state.single_result

        st.markdown("---")

        # Original text
        st.subheader("📄 Original Text")
        st.info(result['text'])

        # Side-by-side comparison
        st.markdown('<div class="sub-header">🔄 Model Comparison</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔵 TextBlob Model")

            tb_data = result['textblob']
            sentiment_class = f"sentiment-{tb_data['sentiment'].lower()}"

            st.markdown(f"""
            <p>Sentiment: <span class="{sentiment_class}">{get_sentiment_emoji(tb_data['sentiment'])} {tb_data['sentiment']}</span></p>
            <p>Polarity: <b>{format_score(tb_data['polarity'])}</b></p>
            <p>Subjectivity: <b>{format_score(tb_data['subjectivity'])}</b></p>
            """, unsafe_allow_html=True)

            st.metric("Sentiment", tb_data['sentiment'])

        with col2:
            st.markdown("### 🟣 VADER Model")

            vader_data = result['vader']
            sentiment_class = f"sentiment-{vader_data['sentiment'].lower()}"

            st.markdown(f"""
            <p>Sentiment: <span class="{sentiment_class}">{get_sentiment_emoji(vader_data['sentiment'])} {vader_data['sentiment']}</span></p>
            <p>Compound Score: <b>{format_score(vader_data['compound'])}</b></p>
            <p>Positive: {format_score(vader_data['pos'])} | Negative: {format_score(vader_data['neg'])} | Neutral: {format_score(vader_data['neu'])}</p>
            """, unsafe_allow_html=True)

            st.metric("Sentiment", vader_data['sentiment'])

        # Mini word cloud
        if len(result['text']) > 20:
            st.markdown("---")
            st.markdown('<div class="sub-header">☁️ Word Cloud</div>', unsafe_allow_html=True)

            wordcloud_fig = create_word_cloud([result['text']], width=1000, height=400)
            st.pyplot(wordcloud_fig)


def tab_bulk_analysis():
    """
    Tab 2: Bulk CSV analysis with file upload and batch processing.
    """
    st.markdown('<div class="sub-header">📊 Bulk CSV Analysis</div>', unsafe_allow_html=True)

    # File upload
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file with feedback data:",
            type=['csv'],
            help="CSV should contain a text column (e.g., 'feedback', 'text', 'comment')"
        )

    with col2:
        st.markdown("**Or use sample data:**")
        use_sample = st.button("📁 Load Sample Data", use_container_width=True)

    # Options
    st.markdown("### ⚙️ Processing Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        remove_duplicates = st.checkbox("Remove duplicate entries", value=True)

    with col2:
        min_length = st.slider("Minimum text length", 3, 100, 10)

    with col3:
        include_neutral = st.checkbox("Include neutral sentiment", value=True)

    # Process button
    if uploaded_file or use_sample:
        if st.button("🚀 Process Data", type="primary", use_container_width=True):
            # Load data
            if use_sample:
                with st.spinner("Loading sample data..."):
                    df = load_sample_data()
                    st.success(f"✅ Loaded {len(df)} sample feedback entries")
            else:
                # Validate file
                is_valid, error_msg = validate_csv_file(uploaded_file)

                if not is_valid:
                    st.error(f"❌ {error_msg}")
                    return

                with st.spinner("Reading CSV file..."):
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded {len(df)} entries from CSV")

            # Get text column
            try:
                text_col = get_text_column_from_csv(df)
            except ValueError as e:
                st.error(f"❌ {str(e)}")
                return

            st.info(f"📌 Using column: '{text_col}' for analysis")

            # Apply filters
            if remove_duplicates:
                original_len = len(df)
                df = df.drop_duplicates(subset=[text_col])
                if len(df) < original_len:
                    st.warning(f"⚠️ Removed {original_len - len(df)} duplicate entries")

            # Filter by text length
            df = df[df[text_col].str.len() >= min_length]

            # Analyze
            with st.spinner("🔍 Analyzing sentiment with both models..."):
                texts = df[text_col].tolist()
                results_df = st.session_state.analyzer.analyze_batch(texts)

                # Add metadata from original dataframe if available
                if len(results_df) == len(df):
                    for col in df.columns:
                        if col != text_col:
                            results_df[col] = df[col].values

                # Filter by sentiment if needed
                if not include_neutral:
                    results_df = results_df[
                        (results_df['textblob_sentiment'] != 'Neutral') &
                        (results_df['vader_sentiment'] != 'Neutral')
                    ]

                st.session_state.analysis_results = results_df
                st.session_state.current_texts = texts
                st.success(f"✅ Analysis complete! Processed {len(results_df)} entries")

    # Display results
    if st.session_state.analysis_results is not None:
        st.markdown("---")
        st.markdown('<div class="sub-header">📈 Analysis Results</div>', unsafe_allow_html=True)

        results_df = st.session_state.analysis_results

        # Summary cards
        summary = create_analysis_summary(results_df)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Analyzed", summary['total_feedback'])

        with col2:
            st.metric("Agreement Rate", f"{summary['agreement_rate']}%")

        with col3:
            st.metric("Avg TextBlob", f"{summary['avg_textblob_score']:.3f}")

        with col4:
            st.metric("Avg VADER", f"{summary['avg_vader_score']:.3f}")

        # Sentiment distribution tabs
        tab1, tab2, tab3 = st.tabs(["📊 Charts", "📋 Data Table", "💾 Export"])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(create_sentiment_pie_chart(results_df, 'textblob'), use_container_width=True)

            with col2:
                st.plotly_chart(create_sentiment_pie_chart(results_df, 'vader'), use_container_width=True)

            st.plotly_chart(create_comparison_bar_graph(results_df), use_container_width=True)

        with tab2:
            st.dataframe(
                results_df,
                column_config={
                    'text': st.column_config.TextColumn('Feedback', width='large'),
                    'textblob_sentiment': st.column_config.TextColumn('TextBlob Sentiment', width='small'),
                    'textblob_polarity': st.column_config.NumberColumn('Polarity', format='%.4f'),
                    'vader_sentiment': st.column_config.TextColumn('VADER Sentiment', width='small'),
                    'vader_compound': st.column_config.NumberColumn('Compound', format='%.4f')
                },
                use_container_width=True,
                height=400
            )

        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                csv_data = export_results(results_df, 'csv')
                st.download_button(
                    "📥 Download as CSV",
                    csv_data,
                    "sentiment_analysis_results.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col2:
                excel_data = export_results(results_df, 'excel')
                st.download_button(
                    "📥 Download as Excel",
                    excel_data,
                    "sentiment_analysis_results.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )


def tab_dashboard():
    """
    Tab 3: Dashboard with comprehensive insights and visualizations.
    """
    st.markdown('<div class="sub-header">🎯 Dashboard & Insights</div>', unsafe_allow_html=True)

    # Check if data is available
    if st.session_state.analysis_results is None:
        st.markdown("""
        <div class="info-box">
        ℹ️ <b>No data available yet.</b> Please go to the "Bulk CSV Analysis" tab
        to load and analyze data first.
        </div>
        """, unsafe_allow_html=True)
        return

    results_df = st.session_state.analysis_results

    # Generate summary
    summary = create_analysis_summary(results_df)
    comparison = st.session_state.analyzer.compare_models(results_df)

    # Overview section
    st.markdown("### 📊 Overview Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Feedback", summary['total_feedback'], delta="Total")

    with col2:
        st.metric("Model Agreement", f"{summary['agreement_rate']}%",
                 delta=f"{comparison['agreement_count']} agreements")

    with col3:
        pos_pct = summary['textblob_percentages'].get('Positive', 0)
        st.metric("Positive (TextBlob)", f"{pos_pct}%")

    with col4:
        neg_pct = summary['textblob_percentages'].get('Negative', 0)
        st.metric("Negative (TextBlob)", f"{neg_pct}%")

    st.markdown("---")

    # Interactive visualizations
    st.markdown('<div class="sub-header">📈 Visualizations</div>', unsafe_allow_html=True)

    # Row 1: Distribution charts
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_sentiment_pie_chart(results_df, 'textblob'), use_container_width=True)

    with col2:
        st.plotly_chart(create_sentiment_pie_chart(results_df, 'vader'), use_container_width=True)

    # Row 2: Comparison and distribution
    st.plotly_chart(create_comparison_bar_graph(results_df), use_container_width=True)
    st.plotly_chart(create_score_distribution_plot(results_df), use_container_width=True)

    # Row 3: Model agreement and scatter
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_model_agreement_chart(comparison), use_container_width=True)

    with col2:
        st.plotly_chart(create_score_comparison_scatter(results_df), use_container_width=True)

    # Word clouds
    st.markdown("---")
    st.markdown('<div class="sub-header">☁️ Word Clouds by Sentiment</div>', unsafe_allow_html=True)

    sentiment_filter = st.selectbox(
        "Filter by sentiment:",
        ["All", "Positive", "Negative", "Neutral"]
    )

    if sentiment_filter == "All":
        filtered_texts = results_df['text'].tolist()
    else:
        filtered_texts = results_df[results_df['textblob_sentiment'] == sentiment_filter]['text'].tolist()

    if filtered_texts:
        wordcloud_fig = create_word_cloud(filtered_texts, width=1200, height=500)
        st.pyplot(wordcloud_fig)
    else:
        st.info("📭 No texts available for selected sentiment")

    # Keywords
    st.markdown("---")
    st.markdown('<div class="sub-header">🔑 Top Keywords</div>', unsafe_allow_html=True)

    keywords = extract_keywords(results_df['text'].tolist(), top_n=15)
    keyword_chart = create_keyword_frequency_chart(keywords, top_n=15)
    st.plotly_chart(keyword_chart, use_container_width=True)

    # Insights panel
    st.markdown("---")
    st.markdown('<div class="sub-header">💡 Key Insights</div>', unsafe_allow_html=True)

    insights_col1, insights_col2 = st.columns(2)

    with insights_col1:
        st.markdown("#### TextBlob Findings")
        st.markdown(f"""
        - **Positive**: {summary['textblob_percentages']['Positive']}% of feedback
        - **Negative**: {summary['textblob_percentages']['Negative']}% of feedback
        - **Neutral**: {summary['textblob_percentages']['Neutral']}% of feedback
        - **Average Polarity**: {summary['avg_textblob_score']:.4f}
        - **Average Subjectivity**: {summary['average_subjectivity']:.3f}
        """)

    with insights_col2:
        st.markdown("#### VADER Findings")
        st.markdown(f"""
        - **Positive**: {summary['vader_percentages']['Positive']}% of feedback
        - **Negative**: {summary['vader_percentages']['Negative']}% of feedback
        - **Neutral**: {summary['vader_percentages']['Neutral']}% of feedback
        - **Average Compound**: {summary['avg_vader_score']:.4f}
        - **Agreement with TextBlob**: {summary['agreement_rate']}%
        """)

    # Model comparison insights
    st.markdown("#### Model Comparison")

    if summary['agreement_rate'] >= 80:
        st.success(f"✅ High agreement ({summary['agreement_rate']:.1f}%) - Both models largely agree on sentiment classification")
    elif summary['agreement_rate'] >= 60:
        st.info(f"ℹ️ Moderate agreement ({summary['agreement_rate']:.1f}%) - Models show some differences in classification")
    else:
        st.warning(f"⚠️ Low agreement ({summary['agreement_rate']:.1f}%) - Models differ significantly in their classifications")

    # Show disagreement cases if any
    if comparison['disagreement_count'] > 0:
        with st.expander(f"🔍 View {comparison['disagreement_count']} Disagreement Cases"):
            disagreement_df = comparison['disagreement_cases'][['text', 'textblob_sentiment', 'vader_sentiment', 'textblob_polarity', 'vader_compound']]
            st.dataframe(disagreement_df, use_container_width=True)


def main():
    """
    Main application function that sets up the UI and manages navigation.
    """
    # Initialize session state
    initialize_session_state()

    # Render header
    render_header()

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📝 Single Analysis",
        "📊 Bulk CSV Analysis",
        "🎯 Dashboard & Insights"
    ])

    with tab1:
        tab_single_analysis()

    with tab2:
        tab_bulk_analysis()

    with tab3:
        tab_dashboard()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    <b>Student Feedback Sentiment Analysis Dashboard</b><br>
    Built with Streamlit | TextBlob & VADER NLP Models<br>
    <small>Analysis generated at {}</small>
    </div>
    """.format(get_timestamp()), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
