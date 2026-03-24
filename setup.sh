#!/bin/bash
# Student Feedback Sentiment Analysis Dashboard - Setup Script

echo "=========================================="
echo "🎓 Student Feedback Sentiment Analysis"
echo "=========================================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "=========================================="
echo "🚀 To run the application:"
echo "=========================================="
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the Streamlit app:"
echo "   streamlit run app.py"
echo ""
echo "The dashboard will open at: http://localhost:8501"
echo ""
