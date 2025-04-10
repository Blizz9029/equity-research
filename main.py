import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import numpy as np
import re
import os
import tempfile
import plotly.express as px
from textblob import TextBlob
import yfinance as yf
import requests
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Equity Research Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6c63ff;
        color: white;
    }
    h1, h2, h3 {
        color: #3a4a65;
    }
    .metric-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin-bottom: 10px;
    }
    .sentiment-positive {
        color: #0f8c40;
        font-weight: 600;
    }
    .sentiment-negative {
        color: #e63946;
        font-weight: 600;
    }
    .sentiment-neutral {
        color: #898989;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class PDFExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = ""
        self.sections = {}
        self.tables = []
        self.metrics = {}
        
    def extract_text(self):
        """Extract full text from PDF using PyMuPDF"""
        doc = fitz.open(self.file_path)
        self.text = ""
        for page in doc:
            self.text += page.get_text()
        doc.close()
        return self.text
    
    def extract_tables(self):
        """Extract tables using pdfplumber"""
        with pdfplumber.open(self.file_path) as pdf:
            self.tables = []
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    self.tables.extend(tables)
        return self.tables
    
    def identify_sections(self):
        """Identify key sections in the research report"""
        # Common section headers in research reports
        section_patterns = {
            'summary': r'(?i)(executive\s+summary|investment\s+summary|summary)',
            'recommendation': r'(?i)(recommendation|rating|investment\s+view)',
            'target_price': r'(?i)(target\s+price|price\s+target)',
            'financials': r'(?i)(financials|financial\s+summary|key\s+financials)',
            'risks': r'(?i)(risks?|risk\s+factors|key\s+risks)',
            'valuation': r'(?i)(valuation|valuation\s+methodology)'
        }
        
        for section_name, pattern in section_patterns.items():
            matches = re.finditer(pattern, self.text)
            for match in matches:
                start_pos = match.start()
                # Find the next section or use end of text
                next_section_pos = len(self.text)
                for other_pattern in section_patterns.values():
                    other_matches = re.finditer(other_pattern, self.text[start_pos + 1:])
                    for other_match in other_matches:
                        other_start = start_pos + 1 + other_match.start()
                        if other_start < next_section_pos:
                            next_section_pos = other_start
                
                self.sections[section_name] = self.text[start_pos:next_section_pos].strip()
                break  # Just use the first match for each section
        
        return self.sections
    
    def extract_metrics(self):
        """Extract key financial metrics"""
        # Extract target price
        target_price_pattern = r'(?i)target\s+price[:\s]+(?:INR|Rs\.?|₹)?[\s]*([0-9,.]+)'
        target_matches = re.search(target_price_pattern, self.text)
        if target_matches:
            self.metrics['target_price'] = target_matches.group(1).replace(',', '')
        
        # Extract recommendation
        rec_pattern = r'(?i)(buy|sell|hold|neutral|overweight|underweight|outperform|market\s+perform|market\s+outperform|reduce)'
        rec_matches = re.search(rec_pattern, self.text[:500])  # Usually at the beginning
        if rec_matches:
            self.metrics['recommendation'] = rec_matches.group(1).upper()
        
        # Extract expected return/upside
        upside_pattern = r'(?i)(?:upside|return)[\s:]+([0-9,.]+)%'
        upside_matches = re.search(upside_pattern, self.text[:1000])
        if upside_matches:
            self.metrics['expected_return'] = upside_matches.group(1)
        
        # Extract company name and ticker
        company_pattern = r'(?i)(?:company|corp|inc|ltd)[\s:]+([A-Za-z0-9\s]+)'
        company_matches = re.search(company_pattern, self.text[:1000])
        if company_matches:
            self.metrics['company'] = company_matches.group(1).strip()
            
        # Try to extract ticker
        ticker_pattern = r'(?i)(?:ticker|symbol)[\s:]+([A-Z]+)'
        ticker_matches = re.search(ticker_pattern, self.text[:1000])
        if ticker_matches:
            self.metrics['ticker'] = ticker_matches.group(1).strip()
        
        return self.metrics
    
    def extract_sentiment(self):
        """Extract sentiment from the text"""
        if not self.text:
            self.extract_text()
            
        # Use TextBlob for sentiment analysis
        blob = TextBlob(self.text)
        sentiment = blob.sentiment
        
        return {
            'polarity': sentiment.polarity,
            'subjectivity': sentiment.subjectivity,
            'sentiment': 'positive' if sentiment.polarity > 0.1 else 'negative' if sentiment.polarity < -0.1 else 'neutral'
        }
    
    def process(self):
        """Run the full extraction process"""
        self.extract_text()
        self.extract_tables()
        self.identify_sections()
        self.extract_metrics()
        sentiment = self.extract_sentiment()
        
        return {
            'metrics': self.metrics,
            'sections': self.sections,
            'tables': self.tables,
            'sentiment': sentiment,
            'text': self.text
        }

def get_stock_news(ticker):
    """Get latest news for a stock ticker"""
    try:
        # Using a free API for news
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey=demo"
        response = requests.get(url)
        data = response.json()
        
        if 'feed' in data:
            return data['feed'][:5]  # Return top 5 news items
        else:
            return []
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def get_stock_data(ticker):
    """Get stock price data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        return hist, stock.info
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame(), {}

def analyze_reports(uploaded_files):
    """Analyze multiple research reports"""
    all_results = []
    
    with st.spinner('Processing PDF files...'):
        for uploaded_file in uploaded_files:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            
            try:
                # Process the PDF
                extractor = PDFExtractor(temp_file_path)
                result = extractor.process()
                
                # Add filename to results
                result['filename'] = uploaded_file.name
                all_results.append(result)
                
            finally:
                # Remove the temp file
                os.unlink(temp_file_path)
    
    return all_results

def display_metrics(results):
    """Display key metrics from the reports"""
    if not results:
        return
        
    # Get metrics from all reports
    all_metrics = [r['metrics'] for r in results]
    
    # Create columns for metrics display
    cols = st.columns(len(results))
    
    for i, (metrics, col) in enumerate(zip(all_metrics, cols)):
        with col:
            st.subheader(f"Report {i+1} Metrics")
            
            # Display recommendation with color
            if 'recommendation' in metrics:
                rec = metrics['recommendation']
                color = "green" if "BUY" in rec or "OVERWEIGHT" in rec or "OUTPERFORM" in rec else \
                       "red" if "SELL" in rec or "UNDERWEIGHT" in rec or "REDUCE" in rec else "orange"
                st.markdown(f"<div class='metric-card'><b>Recommendation:</b> <span style='color:{color};font-weight:bold'>{rec}</span></div>", unsafe_allow_html=True)
            
            # Display target price
            if 'target_price' in metrics:
                st.markdown(f"<div class='metric-card'><b>Target Price:</b> {metrics['target_price']}</div>", unsafe_allow_html=True)
            
            # Display expected return
            if 'expected_return' in metrics:
                st.markdown(f"<div class='metric-card'><b>Expected Return:</b> {metrics['expected_return']}%</div>", unsafe_allow_html=True)
            
            # Display company if available
            if 'company' in metrics:
                st.markdown(f"<div class='metric-card'><b>Company:</b> {metrics['company']}</div>", unsafe_allow_html=True)
            
            # Display ticker if available
            if 'ticker' in metrics:
                st.markdown(f"<div class='metric-card'><b>Ticker:</b> {metrics['ticker']}</div>", unsafe_allow_html=True)
    
    # Display sentiment comparison if we have multiple reports
    if len(results) > 1:
        st.subheader("Sentiment Comparison")
        
        sentiment_data = []
        for i, r in enumerate(results):
            sentiment = r['sentiment']
            sentiment_data.append({
                'Report': f"Report {i+1}",
                'Polarity': sentiment['polarity'],
                'Subjectivity': sentiment['subjectivity']
            })
        
        df = pd.DataFrame(sentiment_data)
        
        # Create bar chart for sentiment comparison
        fig = px.bar(df, x='Report', y='Polarity', color='Polarity',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    title="Sentiment Polarity Comparison")
        st.plotly_chart(fig, use_container_width=True)

def display_sections(results):
    """Display key sections from the reports"""
    if not results:
        return
    
    # Create tabs for each report
    tabs = st.tabs([f"Report {i+1}" for i in range(len(results))])
    
    for i, (tab, result) in enumerate(zip(tabs, results)):
        with tab:
            sections = result['sections']
            
            if not sections:
                st.info("No sections were identified in this report.")
                continue
            
            # Display each section in an expander
            for section_name, content in sections.items():
                with st.expander(f"{section_name.replace('_', ' ').title()}"):
                    st.markdown(content)
            
            # Display sentiment for this report
            sentiment = result['sentiment']
            sentiment_class = "sentiment-positive" if sentiment['polarity'] > 0.1 else \
                             "sentiment-negative" if sentiment['polarity'] < -0.1 else "sentiment-neutral"
            
            st.markdown(f"""
            <div style='margin-top:20px;padding:15px;background-color:#f5f5f5;border-radius:5px;'>
                <h3>Sentiment Analysis</h3>
                <p>Overall sentiment: <span class='{sentiment_class}'>{sentiment['sentiment'].title()}</span></p>
                <p>Polarity: {sentiment['polarity']:.2f} (range -1 to 1, negative to positive)</p>
                <p>Subjectivity: {sentiment['subjectivity']:.2f} (range 0 to 1, objective to subjective)</p>
            </div>
            """, unsafe_allow_html=True)

def fetch_stock_info(results):
    """Fetch and display stock information and news"""
    # Try to find a ticker in any of the reports
    ticker = None
    for result in results:
        if 'ticker' in result['metrics']:
            ticker = result['metrics']['ticker']
            break
    
    if not ticker:
        # Try to extract from company name
        for result in results:
            if 'company' in result['metrics']:
                company = result['metrics']['company']
                st.subheader(f"Company: {company}")
                st.warning("No ticker symbol found. Please enter it manually to fetch market data.")
                ticker = st.text_input("Enter Ticker Symbol", "")
                break
    
    if ticker:
        st.subheader(f"Market Data for {ticker}")
        cols = st.columns([2, 1])
        
        with cols[0]:
            # Get stock price history
            hist, info = get_stock_data(ticker)
            if not hist.empty:
                # Plot stock price
                fig = px.line(hist, y='Close', title=f"{ticker} Stock Price (Last Month)")
                fig.update_layout(xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Could not fetch price data for {ticker}")
        
        with cols[1]:
            # Display basic stock info
            if info:
                st.markdown("<h4>Stock Info</h4>", unsafe_allow_html=True)
                metrics = {
                    "Current Price": info.get('currentPrice', 'N/A'),
                    "Market Cap": f"{info.get('marketCap', 0) / 1e9:.2f}B" if 'marketCap' in info else 'N/A',
                    "52-Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
                    "52-Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
                    "P/E Ratio": info.get('trailingPE', 'N/A'),
                    "Dividend Yield": f"{info.get('dividendYield', 0) * 100:.2f}%" if 'dividendYield' in info else 'N/A'
                }
                
                for key, value in metrics.items():
                    st.markdown(f"<div class='metric-card'><b>{key}:</b> {value}</div>", unsafe_allow_html=True)
        
        # Get latest news
        st.subheader("Latest News")
        news = get_stock_news(ticker)
        
        if news:
            for item in news:
                with st.expander(f"{item.get('title', 'News Item')}"):
                    st.write(f"**Source:** {item.get('source', 'Unknown')}")
                    st.write(f"**Published:** {item.get('time_published', 'Unknown')}")
                    st.write(item.get('summary', 'No summary available.'))
                    if 'url' in item:
                        st.markdown(f"[Read more]({item['url']})")
        else:
            st.info("No recent news found.")

def main():
    st.title("📊 Equity Research Report Analyzer")
    
    st.markdown("""
    Upload PDF research reports to extract key insights, metrics, and perform comparative analysis.
    The tool extracts target prices, recommendations, key sections, and performs sentiment analysis.
    """)
    
    uploaded_files = st.file_uploader("Upload Research Reports (PDF)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        # Process the uploaded files
        results = analyze_reports(uploaded_files)
        
        if results:
            # Display metrics
            st.header("Key Metrics")
            display_metrics(results)
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Report Sections", "Market Data", "Raw Text"])
            
            with tab1:
                display_sections(results)
            
            with tab2:
                fetch_stock_info(results)
            
            with tab3:
                # Display raw text for debugging
                report_selection = st.selectbox("Select Report", [f"Report {i+1}" for i in range(len(results))])
                report_index = int(report_selection.split()[-1]) - 1
                
                if 0 <= report_index < len(results):
                    st.text_area("Raw Text", results[report_index]['text'], height=400)
    else:
        # Display sample images to show what the app can do
        st.info("👆 Upload PDF research reports to start the analysis")
        
        # Demo section
        st.header("What this tool can do")
        
        demo_cols = st.columns(3)
        with demo_cols[0]:
            st.markdown("### 📈 Extract Key Metrics")
            st.markdown("• Target prices\n• Recommendations\n• Expected returns\n• Financial data")
            
        with demo_cols[1]:
            st.markdown("### 🔍 Analyze Sentiment")
            st.markdown("• Positive/negative sentiment\n• Objective vs. subjective language\n• Compare multiple reports")
            
        with demo_cols[2]:
            st.markdown("### 📰 Latest Market Data")
            st.markdown("• Current stock prices\n• Stock performance\n• Recent news\n• Key financial ratios")

if __name__ == "__main__":
    main()
