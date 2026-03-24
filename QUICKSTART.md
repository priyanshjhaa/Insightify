# 🚀 Quick Start Guide

## For macOS/Linux Users

### Option 1: Automated Setup (Recommended)

```bash
cd /Users/priyanshjha/NLP\ PROJECT/
./setup.sh
```

Then run the app:
```bash
source venv/bin/activate
streamlit run app.py
```

### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data:**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## First Time Usage

1. The dashboard will open at `http://localhost:8501`

2. **Try Single Analysis first:**
   - Go to Tab 1: "📝 Single Analysis"
   - Enter: "The course content was excellent and very well organized!"
   - Click "🔍 Analyze"
   - View the results from both models

3. **Try Bulk Analysis:**
   - Go to Tab 2: "📊 Bulk CSV Analysis"
   - Click "📁 Load Sample Data"
   - Click "🚀 Process Data"
   - Explore the results and visualizations

4. **View Dashboard:**
   - Go to Tab 3: "🎯 Dashboard & Insights"
   - Explore all the charts and insights

## Sample Data Included

The project includes `data/sample_feedback.csv` with 50 diverse student feedback entries covering:
- Positive feedback (40%)
- Negative feedback (35%)
- Neutral feedback (25%)
- Various categories: Content Quality, Instructor, Assignments, Pace, etc.

## Troubleshooting

### Issue: Module not found
**Solution:** Make sure you activated the virtual environment:
```bash
source venv/bin/activate
```

### Issue: NLTK data error
**Solution:** Download NLTK data manually:
```bash
python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('brown')
>>> exit()
```

### Issue: Streamlit won't start
**Solution:** Try with specific port:
```bash
streamlit run app.py --server.port 8501
```

### Issue: Display issues on macOS
**Solution:** If using macOS with Retina display, set environment variable:
```bash
export STREAMLIT_SERVER_HEADLESS=true
streamlit run app.py
```

## Project Features

✅ **3 Interactive Tabs:**
- Single text analysis
- Bulk CSV processing
- Comprehensive dashboard

✅ **2 NLP Models:**
- TextBlob (lexicon-based)
- VADER (social-media-optimized)

✅ **Multiple Visualizations:**
- Pie charts
- Bar graphs
- Score distributions
- Word clouds
- Keyword analysis

✅ **Export Options:**
- CSV download
- Excel download

## Next Steps

1. **Upload your own CSV:**
   - Prepare a CSV with a feedback column
   - Use Tab 2 to upload and analyze

2. **Explore the insights:**
   - Compare model performance
   - Identify trends
   - Extract key themes

3. **Export results:**
   - Download analyzed data
   - Use for reports or further analysis

## Academic Presentation Tips

When presenting this project:

1. **Start with Single Analysis tab**
   - Show real-time analysis
   - Explain both models briefly

2. **Move to Bulk Analysis tab**
   - Demonstrate CSV upload
   - Show processing options

3. **End with Dashboard tab**
   - Highlight visualizations
   - Discuss model comparison
   - Explain key insights

## Support

For detailed information, see [README.md](README.md)

---

**Happy Analyzing! 🎓**
