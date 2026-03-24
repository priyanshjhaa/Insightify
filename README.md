# 🎓 Student Feedback Sentiment Analysis Dashboard

A comprehensive web-based NLP application that analyzes student feedback using two different sentiment analysis models — **TextBlob** and **VADER** — with comparative visualizations and actionable insights.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Academic Context](#academic-context)
- [Future Enhancements](#future-enhancements)
- [References](#references)

---

## 🎯 Overview

This dashboard is designed for educational institutions to analyze student feedback efficiently. It leverages Natural Language Processing (NLP) to classify feedback sentiment and provides comparative analysis between two popular sentiment analysis models.

**Key Capabilities:**
- Sentence-level sentiment analysis
- Bulk CSV file processing
- Interactive visualizations (charts, word clouds)
- Comparative model insights
- Export functionality for further analysis

---

## ✨ Features

### 1. Single Text Analysis
- Input individual feedback texts
- Real-time sentiment classification
- Side-by-side model comparison
- Instant word cloud generation

### 2. Bulk CSV Analysis
- Upload and process multiple feedback entries
- Configurable processing options (remove duplicates, filter by length)
- Progress tracking during analysis
- Export results to CSV/Excel formats

### 3. Dashboard & Insights
- Comprehensive statistics and metrics
- Multiple visualization types:
  - Sentiment distribution pie charts
  - Comparative bar graphs
  - Score distribution histograms
  - Model agreement analysis
  - Word clouds by sentiment type
  - Keyword frequency analysis
- Interactive filtering options
- Detailed insights panel

---

## 🎯 Objectives

- **Classification**: Categorize student feedback into Positive, Negative, or Neutral sentiment
- **Comparison**: Compare results from TextBlob and VADER models
- **Visualization**: Present sentiment distribution through interactive graphs
- **Insight Extraction**: Derive meaningful insights from textual data

---

## 🏗️ System Architecture

```
User Input (Text/CSV)
    ↓
Preprocessing (cleaning, validation)
    ↓
Sentiment Analysis (TextBlob + VADER)
    ↓
Classification (>0 Positive, <0 Negative, =0 Neutral)
    ↓
Comparison & Statistics
    ↓
Visualization (Charts, Word Clouds)
    ↓
Presentation (Streamlit Multi-tab Dashboard)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone/Download the Project

```bash
cd /path/to/your/projects
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Tab 1: Single Text Analysis

1. **Enter Text**: Type or paste feedback into the text area (3-5000 characters)
2. **Click Analyze**: Press the "🔍 Analyze" button
3. **View Results**:
   - See side-by-side comparison of TextBlob and VADER results
   - View polarity/compound scores
   - See sentiment classification with visual indicators
   - Explore the word cloud for key terms

### Tab 2: Bulk CSV Analysis

1. **Upload CSV**: Click "Browse files" to upload your CSV file
   - CSV must contain a text column (e.g., 'feedback', 'text', 'comment')
2. **Or Use Sample Data**: Click "📁 Load Sample Data" to try the built-in dataset
3. **Configure Options**:
   - Remove duplicate entries
   - Set minimum text length
   - Include/exclude neutral sentiment
4. **Process**: Click "🚀 Process Data" to analyze
5. **Explore Results**:
   - View summary statistics
   - Browse charts and visualizations
   - Filter data table
   - Export results as CSV/Excel

### Tab 3: Dashboard & Insights

1. **Load Data First**: Process data in Tab 2 before accessing the Dashboard
2. **View Overview**: See total analyzed, agreement rate, sentiment percentages
3. **Explore Visualizations**:
   - Pie charts for sentiment distribution
   - Comparative bar graphs
   - Score distribution plots
   - Model agreement charts
   - Scatter plot comparing model scores
4. **Word Clouds**: Filter by sentiment type to see key terms
5. **Keywords**: View top 15 most frequent words in feedback
6. **Read Insights**: Review key findings and model comparison notes

---

## 🔬 Methodology

### Sentiment Analysis Models

#### 1. TextBlob (Lexicon-based Approach)

**How It Works:**
- Uses a pre-built lexicon (dictionary) of words with polarity scores
- Calculates polarity by averaging the scores of all words in text
- Returns polarity score (-1 to +1) and subjectivity score (0 to 1)

**Strengths:**
- Good for longer, well-structured texts
- Provides subjectivity analysis
- Handles context reasonably well

**Limitations:**
- May miss sentiment in informal language
- Struggles with sarcasm and negation
- Fixed lexicon may not include domain-specific terms

#### 2. VADER (Valence Aware Dictionary and sEntiment Reasoner)

**How It Works:**
- Specifically tuned for social media and short texts
- Rule-based approach with heuristics for:
  - Capitalization (e.g., "GOOD" vs "good")
  - Punctuation (e.g., "Good!!!")
  - Emoticons and emojis
  - Negations and intensifiers
- Returns compound score (-1 to +1) and individual positive/negative/neutral proportions

**Strengths:**
- Excellent for short, informal text
- Handles social media language well
- Considers emphasis and punctuation

**Limitations:**
- May not perform as well on longer, formal texts
- Can be overly sensitive to emphasis markers

### Classification Logic

Both models use the same threshold classification:

| Score Range | Sentiment  |
|-------------|------------|
| > 0         | Positive   |
| < 0         | Negative   |
| = 0         | Neutral    |

### Model Comparison

The dashboard tracks:
- **Agreement Rate**: Percentage of texts where both models classify the same sentiment
- **Score Correlation**: How closely the polarity/compound scores relate
- **Disagreement Cases**: Shows specific feedback where models differ

---

## 📁 Project Structure

```
NLP PROJECT/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── sentiment_analyzer.py       # Core: TextBlob + VADER analysis
│   ├── text_processor.py           # Text preprocessing & keywords
│   ├── visualizer.py               # Chart generation
│   └── utils.py                    # Helper functions
│
├── data/
│   └── sample_feedback.csv         # Sample dataset (50 entries)
│
└── outputs/                        # Auto-generated outputs
    └── processed_results/
```

---

## 🔧 Technical Details

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.31.0 | Web framework |
| textblob | 0.17.1 | Polarity/subjectivity scoring |
| vaderSentiment | 3.3.2 | Social-media-optimized scoring |
| pandas | 2.1.4 | Data manipulation |
| plotly | 5.18.0 | Interactive charts |
| wordcloud | 1.9.3 | Word cloud generation |
| matplotlib | 3.8.2 | Plotting backend |
| nltk | 3.8.1 | Tokenization |

### Color Scheme

- **Positive**: Green (#2ecc71)
- **Negative**: Red (#e74c3c)
- **Neutral**: Gray (#95a5a6)

### File Validation

The CSV validator checks for:
- Correct file format (.csv)
- File size limit (10MB maximum)
- Presence of text column
- Non-empty content

Supported column names: `feedback`, `text`, `comment`, `message`, `student_feedback`, `review`, `response`, `input`, `content`

---

## 🎓 Academic Context

### Course Project

This project was developed for a Natural Language Processing (NLP) course to demonstrate:

1. **Understanding of NLP Models**: Comparative analysis of different approaches
2. **Practical Application**: Real-world use case in educational analytics
3. **Data Visualization**: Presenting NLP results effectively
4. **Software Engineering**: Clean, modular code architecture

### Learning Objectives

- Implement and compare sentiment analysis models
- Design user-friendly NLP applications
- Handle real-world data processing challenges
- Create compelling visualizations for insights

### Use Cases

- **Course Evaluation**: Analyze end-of-semester feedback
- **Instructor Assessment**: Understand student perceptions
- **Content Review**: Identify strengths and weaknesses in course material
- **Trend Analysis**: Track sentiment over time

---

## 🚀 Future Enhancements

Potential improvements for future versions:

1. **Additional Models**
   - BERT-based sentiment analysis
   - RoBERTa for more accurate classification
   - Domain-specific models

2. **Advanced Features**
   - Multi-language support
   - Aspect-based sentiment (e.g., analyze instructor vs content separately)
   - Emotion detection (joy, anger, fear, etc.)
   - Trend analysis over time

3. **User Interface**
   - Dark mode option
   - Customizable color schemes
   - Mobile-responsive design
   - Real-time streaming analysis

4. **Data Management**
   - Database integration for storing historical analyses
   - User authentication and saved analyses
   - API endpoints for external access

5. **Reporting**
   - PDF report generation
   - Automated insights summary
   - Presentation mode for demos

---

## 📚 References

### Academic Resources

- **TextBlob Documentation**: https://textblob.readthedocs.io/
- **VADER Paper**: Hutto, C.J., & Gilbert, E.E. (2014). "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text"
- **NLTK Documentation**: https://www.nltk.org/
- **Streamlit Documentation**: https://docs.streamlit.io/

### Sentiment Analysis

- Liu, B. (2012). "Sentiment Analysis and Opinion Mining"
- Pang, B., & Lee, L. (2008). "Opinion Mining and Sentiment Analysis"
- Feldman, R. (2013). "Techniques and Applications for Sentiment Analysis"

---

## 👨‍💻 Development

### Code Quality

- **Docstrings**: Google-style documentation for all functions
- **Type Hints**: Python type hints throughout
- **PEP 8 Compliance**: Follows Python style guide
- **Error Handling**: Comprehensive validation and error messages

### Testing

The sample data includes diverse scenarios:
- Positive feedback (40%)
- Negative feedback (35%)
- Neutral/mixed feedback (25%)
- Edge cases (very short, very long, emojis)

---

## 📄 License

This project is developed for educational purposes.

---

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the development team.

---

## 📞 Support

For issues or questions:
1. Check the Usage Guide section above
2. Review error messages carefully
3. Ensure all dependencies are installed correctly
4. Verify CSV file format matches requirements

---

**Version**: 1.0.0
**Last Updated**: 2024
**Developed for**: Natural Language Processing Course Project
