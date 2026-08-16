#!/bin/bash

# Streamlit App Launcher for ANAVID Queue Intelligence System
# Linux/Mac shell script

echo ""
echo "====================================="
echo "  ANAVID Streamlit App Launcher"
echo "====================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from python.org"
    exit 1
fi

echo "[OK] Python found: $(python3 --version)"
echo ""

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "[INFO] Streamlit not found, installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

echo "[OK] Dependencies ready"
echo ""

# Create necessary directories
mkdir -p data/input
mkdir -p data/output/segments
mkdir -p results

echo "[OK] Directories created"
echo ""

# Launch Streamlit app
echo "Launching Streamlit app on http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m streamlit run streamlit_app.py --logger.level=info --client.maxMessageSize=200
