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

# Custom CSS for modern, visually appealing UI
st.markdown("""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap');

    /* Root variables for consistent theming - Beige & White Theme */
    :root {
        --primary-gradient: linear-gradient(135deg, #c9a86c 0%, #8b7355 100%);
        --success-gradient: linear-gradient(135deg, #d4a574 0%, #c9a86c 100%);
        --warning-gradient: linear-gradient(135deg, #e8d5b7 0%, #d4c4a8 100%);
        --info-gradient: linear-gradient(135deg, #f5f0e6 0%, #ebe5d9 100%);
        --card-bg: rgba(255, 255, 255, 0.98);
        --card-shadow: 0 8px 32px 0 rgba(139, 115, 85, 0.12);
        --text-primary: #3d342b;
        --text-secondary: #6b5d52;
        --beige-light: #f9f6f0;
        --beige-medium: #e8dcc8;
        --beige-dark: #c9a86c;
        --brown-accent: #8b7355;
    }

    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #faf8f3 0%, #f5f0e6 100%);
        padding: 0;
    }

    /* Header styling with gradient */
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 2rem 0 1.5rem 0;
        margin-bottom: 1.5rem;
        position: relative;
        animation: fadeInDown 0.8s ease-out;
    }

    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 200px;
        height: 4px;
        background: var(--primary-gradient);
        border-radius: 2px;
    }

    /* Sub-header styling */
    .sub-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-left: 1rem;
        border-left: 5px solid #c9a86c;
        animation: slideInLeft 0.6s ease-out;
    }

    /* Metric card with glassmorphism */
    .metric-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: var(--card-shadow);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
    }

    /* Sentiment coloring */
    .sentiment-positive {
        color: #10b981;
        font-weight: 700;
        font-size: 1.1em;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }

    .sentiment-negative {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.1em;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }

    .sentiment-neutral {
        color: #6b7280;
        font-weight: 700;
        font-size: 1.1em;
    }

    /* Info box with gradient border */
    .info-box {
        background: linear-gradient(135deg, rgba(249, 246, 240, 0.9) 0%, rgba(245, 240, 230, 0.9) 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #c9a86c;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(139, 115, 85, 0.1);
        transition: all 0.3s ease;
    }

    .info-box:hover {
        box-shadow: 0 8px 24px rgba(139, 115, 85, 0.15);
        transform: translateX(5px);
    }

    /* Enhanced button styling */
    .stButton > button {
        background: var(--primary-gradient) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 0.75rem !important;
        box-shadow: 0 4px 16px rgba(139, 115, 85, 0.25) !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(139, 115, 85, 0.35) !important;
    }

    /* Input field styling */
    .stTextArea > div > div > textarea {
        border-radius: 0.75rem !important;
        border: 2px solid #e8dcc8 !important;
        background-color: white !important;
        color: #3d342b !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #c9a86c !important;
        box-shadow: 0 0 0 3px rgba(201, 168, 108, 0.1) !important;
    }

    .stTextArea > div > div > textarea::placeholder {
        color: #9a8b7a !important;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    [data-testid="stMetricDelta"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 8px;
        background: white;
        border-radius: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 0.75rem;
        padding: 12px 24px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient) !important;
        color: white !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #faf8f3 0%, #f5f0e6 100%);
    }

    /* Footer styling */
    footer {
        background: var(--primary-gradient);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-top: 3rem;
        text-align: center;
    }

    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.8;
        }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #8b7355;
    }

    /* Success/Error/Warning messages */
    .success-message {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border-left: 5px solid #10b981;
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }

    .warning-message {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(245, 158, 11, 0.1) 100%);
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }

    /* Data table styling */
    [data-testid="stDataFrame"] {
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }

    /* Plotly chart styling */
    .js-plotly-plot {
        background: white !important;
        border-radius: 1rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(201, 168, 108, 0.12) 0%, rgba(139, 115, 85, 0.12) 100%);
        border-radius: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(201, 168, 108, 0.2);
        color: #8b7355;
    }

    .streamlit-expanderContent {
        background: white;
        border-radius: 0.75rem;
        padding: 1rem;
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

    # Enhanced info box with icons and better styling
    st.markdown("""
    <div class="info-box" style="position: relative; overflow: hidden;">
        <div style="position: absolute; top: -50px; right: -50px; width: 150px; height: 150px; background: linear-gradient(135deg, rgba(201, 168, 108, 0.15), rgba(139, 115, 85, 0.15)); border-radius: 50%;"></div>
        <div style="position: relative; z-index: 1;">
            <h3 style="color: #8b7355; margin-bottom: 0.5rem; font-family: 'Poppins', sans-serif;">✨ Powerful NLP Analytics</h3>
            <p style="color: #6b5d52; line-height: 1.6; margin-bottom: 0;">
                This advanced dashboard analyzes student feedback using two state-of-the-art NLP models:
                <br><br>
                <span style="display: inline-block; background: linear-gradient(135deg, #c9a86c, #8b7355); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;">🔵 TextBlob</span> — Lexicon-based sentiment analysis
                <br>
                <span style="display: inline-block; background: linear-gradient(135deg, #c9a86c, #8b7355); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;">🟣 VADER</span> — Optimized for social media text
                <br><br>
                Compare results, visualize distributions, and extract actionable insights from your feedback data.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def tab_single_analysis():
    """
    Tab 1: Single text analysis with real-time results.
    """
    st.markdown('<div class="sub-header">📝 Single Text Analysis</div>', unsafe_allow_html=True)

    # Text input area with enhanced styling
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(139, 115, 85, 0.08);">
        <label style="font-weight: 600; color: #8b7355; margin-bottom: 0.5rem; display: block;">Enter student feedback to analyze:</label>
    </div>
    """, unsafe_allow_html=True)

    text_input = st.text_area(
        "feedback_text",
        label_visibility="collapsed",
        placeholder="Type or paste the feedback text here... (3-5000 characters)",
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
        with st.spinner("🔍 Analyzing sentiment..."):
            result = st.session_state.analyzer.analyze_single_text(text_input)
            st.session_state.single_result = result

    # Display results
    if 'single_result' in st.session_state and st.session_state.single_result:
        result = st.session_state.single_result

        st.markdown("---")

        # Original text with better styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(249, 246, 240, 0.8), rgba(245, 240, 230, 0.8)); padding: 1.5rem; border-radius: 1rem; border-left: 5px solid #c9a86c;">
            <h4 style="color: #8b7355; margin-bottom: 0.75rem;">📄 Original Text</h4>
            <p style="color: #3d342b; line-height: 1.6; margin: 0;">
        """, unsafe_allow_html=True)

        st.markdown(f"<span>{result['text']}</span></p></div>", unsafe_allow_html=True)

        # Side-by-side comparison with enhanced cards
        st.markdown('<div class="sub-header">🔄 Model Comparison</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # TextBlob card
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(201, 168, 108, 0.12), rgba(139, 115, 85, 0.12)); padding: 1.5rem; border-radius: 1rem; border: 2px solid rgba(201, 168, 108, 0.25); backdrop-filter: blur(10px);">
                <h3 style="color: #8b7355; margin-bottom: 1rem;">🔵 TextBlob Model</h3>
            """, unsafe_allow_html=True)

            tb_data = result['textblob']
            sentiment_class = f"sentiment-{tb_data['sentiment'].lower()}"

            st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 0.75rem; margin-top: 1rem;">
                    <p style="margin: 0.5rem 0; color: #6b5d52;">Sentiment:</p>
                    <p class="{sentiment_class}" style="font-size: 1.3rem; margin: 0.5rem 0;">
                        {get_sentiment_emoji(tb_data['sentiment'])} {tb_data['sentiment']}
                    </p>
                    <hr style="border: none; border-top: 1px solid #e8dcc8; margin: 1rem 0;">
                    <p style="margin: 0.5rem 0; color: #6b5d52;">
                        <strong>Polarity:</strong> <span style="color: #c9a86c;">{format_score(tb_data['polarity'])}</span>
                    </p>
                    <p style="margin: 0.5rem 0; color: #6b5d52;">
                        <strong>Subjectivity:</strong> <span style="color: #c9a86c;">{format_score(tb_data['subjectivity'])}</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            st.metric("Overall Sentiment", tb_data['sentiment'],
                     delta_color="normal" if tb_data['sentiment'] == "Positive" else "inverse" if tb_data['sentiment'] == "Negative" else "off")

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # VADER card
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(212, 165, 116, 0.12), rgba(201, 168, 108, 0.12)); padding: 1.5rem; border-radius: 1rem; border: 2px solid rgba(212, 165, 116, 0.25); backdrop-filter: blur(10px);">
                <h3 style="color: #a68b6e; margin-bottom: 1rem;">🟣 VADER Model</h3>
            """, unsafe_allow_html=True)

            vader_data = result['vader']
            sentiment_class = f"sentiment-{vader_data['sentiment'].lower()}"

            st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 0.75rem; margin-top: 1rem;">
                    <p style="margin: 0.5rem 0; color: #6b5d52;">Sentiment:</p>
                    <p class="{sentiment_class}" style="font-size: 1.3rem; margin: 0.5rem 0;">
                        {get_sentiment_emoji(vader_data['sentiment'])} {vader_data['sentiment']}
                    </p>
                    <hr style="border: none; border-top: 1px solid #e8dcc8; margin: 1rem 0;">
                    <p style="margin: 0.5rem 0; color: #6b5d52;">
                        <strong>Compound Score:</strong> <span style="color: #d4a574;">{format_score(vader_data['compound'])}</span>
                    </p>
                    <p style="margin: 0.5rem 0; color: #6b5d52; font-size: 0.9rem;">
                        Positive: <span style="color: #10b981;">{format_score(vader_data['pos'])}</span> |
                        Negative: <span style="color: #ef4444;">{format_score(vader_data['neg'])}</span> |
                        Neutral: <span style="color: #6b7280;">{format_score(vader_data['neu'])}</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            st.metric("Overall Sentiment", vader_data['sentiment'],
                     delta_color="normal" if vader_data['sentiment'] == "Positive" else "inverse" if vader_data['sentiment'] == "Negative" else "off")

            st.markdown("</div>", unsafe_allow_html=True)

        # Mini word cloud with enhanced container
        if len(result['text']) > 20:
            st.markdown("---")
            st.markdown('<div class="sub-header">☁️ Word Cloud Visualization</div>', unsafe_allow_html=True)

            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 1rem; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);">
            """, unsafe_allow_html=True)

            wordcloud_fig = create_word_cloud([result['text']], width=1000, height=400)
            st.pyplot(wordcloud_fig)

            st.markdown("</div>", unsafe_allow_html=True)


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
    <div style="background: linear-gradient(135deg, #c9a86c 0%, #8b7355 100%); color: white; padding: 2.5rem 2rem; border-radius: 1rem; margin-top: 3rem; text-align: center; box-shadow: 0 8px 32px rgba(139, 115, 85, 0.25);">
        <h3 style="margin: 0 0 0.5rem 0; font-family: 'Poppins', sans-serif; font-size: 1.5rem;">Student Feedback Sentiment Analysis Dashboard</h3>
        <p style="margin: 0.5rem 0; opacity: 0.95;">
            Built with ❤️ using Streamlit | TextBlob & VADER NLP Models
        </p>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.85; font-size: 0.9rem;">
            Analysis generated at {}
        </p>
        <div style="margin-top: 1rem; opacity: 0.7;">
            <span style="display: inline-block; margin: 0 0.5rem;">🎓</span>
            <span style="display: inline-block; margin: 0 0.5rem;">📊</span>
            <span style="display: inline-block; margin: 0 0.5rem;">🔍</span>
        </div>
    </div>
    """.format(get_timestamp()), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
