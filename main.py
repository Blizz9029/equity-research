
import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import numpy as np
import re
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
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
    .financial-table {
        font-size: 0.9rem;
    }
    .highlight-row {
        background-color: #f0f8ff;
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
        self.financial_data = {}
        
    def extract_text(self):
        """Extract full text from PDF using PyMuPDF"""
        doc = fitz.open(self.file_path)
        self.text = ""
        for page in doc:
            self.text += page.get_text()
        doc.close()
        return self.text
    
    def extract_tables(self):
        """Extract tables using pdfplumber with improved processing"""
        with pdfplumber.open(self.file_path) as pdf:
            all_tables = []
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        # Clean table data
                        cleaned_table = []
                        for row in table:
                            cleaned_row = [cell.strip() if cell else '' for cell in row]
                            cleaned_table.append(cleaned_row)
                        
                        # Add page number and try to extract table title
                        page_text = page.extract_text()
                        table_title = self._find_table_title(page_text)
                        
                        all_tables.append({
                            'data': cleaned_table,
                            'page': page_num + 1,
                            'title': table_title
                        })
            
            self.tables = all_tables
            return self.tables
    
    def _find_table_title(self, page_text):
        """Try to find a table title in the page text"""
        title_patterns = [
            r'(?:Table|Exhibit|Figure)[ \t]+\d+[:. ]+([^\n]+)',
            r'(?:Table|Exhibit|Figure)[ \t]*:[ \t]*([^\n]+)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, page_text)
            if match:
                return match.group(1).strip()
        
        return "Untitled Table"
    
    def identify_sections(self):
        """Identify key sections in the research report"""
        # Common section headers in research reports
        section_patterns = {
            'summary': r'(?i)(executive\s+summary|investment\s+summary|summary)',
            'recommendation': r'(?i)(recommendation|rating|investment\s+view)',
            'target_price': r'(?i)(target\s+price|price\s+target)',
            'financials': r'(?i)(financials|financial\s+summary|key\s+financials)',
            'risks': r'(?i)(risks?|risk\s+factors|key\s+risks)',
            'valuation': r'(?i)(valuation|valuation\s+methodology)',
            'sector_outlook': r'(?i)(sector\s+outlook|industry\s+outlook)',
            'company_overview': r'(?i)(company\s+overview|company\s+profile|business\s+overview)'
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
        """Extract key financial metrics with improved recognition"""
        # Extract target price
        target_price_patterns = [
            r'(?i)target\s+price[:\s]+(?:INR|Rs\.?|₹)?[\s]*([0-9,.]+)',
            r'(?i)TP\s+(?:of|:)\s+(?:INR|Rs\.?|₹)?[\s]*([0-9,.]+)',
            r'(?i)price\s+target[:\s]+(?:INR|Rs\.?|₹)?[\s]*([0-9,.]+)'
        ]
        
        for pattern in target_price_patterns:
            match = re.search(pattern, self.text)
            if match:
                self.metrics['target_price'] = match.group(1).replace(',', '')
                break
        
        # Extract recommendation
        rec_patterns = [
            r'(?i)(buy|sell|hold|neutral|overweight|underweight|outperform|market\s+perform|market\s+outperform|reduce)',
            r'(?i)recommendation[:\s]+(buy|sell|hold|neutral|overweight|underweight|outperform|market\s+perform|reduce)'
        ]
        
        for pattern in rec_patterns:
            match = re.search(pattern, self.text[:1000])  # Usually at the beginning
            if match:
                self.metrics['recommendation'] = match.group(1).upper()
                break
        
        # Extract expected return/upside
        upside_patterns = [
            r'(?i)(?:upside|return)[:\s]+([0-9,.]+)%',
            r'(?i)(?:upside|return)[ \t]+potential[ \t]+of[ \t]+([0-9,.]+)%',
            r'(?i)potential[ \t]+(?:upside|return)[ \t]+of[ \t]+([0-9,.]+)%'
        ]
        
        for pattern in upside_patterns:
            match = re.search(pattern, self.text[:2000])
            if match:
                self.metrics['expected_return'] = match.group(1)
                break
        
        # Extract company name and industry/sector
        company_pattern = r'(?i)(?:company|corp|inc|ltd)[\s:]+([A-Za-z0-9\s]+)'
        company_match = re.search(company_pattern, self.text[:1000])
        if company_match:
            self.metrics['company'] = company_match.group(1).strip()
        
        sector_pattern = r'(?i)(?:sector|industry)[:\s]+([A-Za-z0-9\s&]+)'
        sector_match = re.search(sector_pattern, self.text[:2000])
        if sector_match:
            self.metrics['sector'] = sector_match.group(1).strip()
            
        # Extract key financial ratios
        pe_pattern = r'(?i)(?:P/E|PE ratio|Price/Earnings)[:\s]+([0-9,.]+)x?'
        pe_match = re.search(pe_pattern, self.text)
        if pe_match:
            self.metrics['pe_ratio'] = pe_match.group(1).replace(',', '')
        
        pb_pattern = r'(?i)(?:P/B|PB ratio|Price/Book)[:\s]+([0-9,.]+)x?'
        pb_match = re.search(pb_pattern, self.text)
        if pb_match:
            self.metrics['pb_ratio'] = pb_match.group(1).replace(',', '')
        
        # Extract ticker symbol - search for common stock exchange patterns
        ticker_patterns = [
            r'(?i)(?:ticker|symbol)[:\s]+([A-Z]+)',
            r'(?i)(?:NSE|BSE|NYSE|NASDAQ)[:\s]+([A-Z]+)',
            r'(?i)\((?:NSE|BSE|NYSE|NASDAQ)[:\s]+([A-Z]+)\)'
        ]
        
        for pattern in ticker_patterns:
            match = re.search(pattern, self.text[:2000])
            if match:
                self.metrics['ticker'] = match.group(1).strip()
                break
        
        return self.metrics
    
    def extract_financial_data(self):
        """Extract key financial data from tables and text"""
        financial_data = {}
        
        # Look for revenue and profit figures in text
        revenue_pattern = r'(?i)revenue[s]?[ \t]+(?:of[ \t]+)?(?:INR|Rs\.?|₹)?[ \t]*([0-9,.]+)[ \t]*(?:Cr|mn|bn)'
        revenue_match = re.search(revenue_pattern, self.text)
        if revenue_match:
            financial_data['revenue'] = revenue_match.group(1).replace(',', '')
        
        profit_pattern = r'(?i)(?:net[ \t]+profit|PAT)[ \t]+(?:of[ \t]+)?(?:INR|Rs\.?|₹)?[ \t]*([0-9,.]+)[ \t]*(?:Cr|mn|bn)'
        profit_match = re.search(profit_pattern, self.text)
        if profit_match:
            financial_data['net_profit'] = profit_match.group(1).replace(',', '')
        
        # Extract growth percentages
        growth_pattern = r'(?i)(?:growth|increase|rise)[ \t]+of[ \t]+([0-9,.]+)%'
        growth_matches = re.finditer(growth_pattern, self.text)
        growth_data = []
        for match in growth_matches:
            # Try to get context (what is growing)
            context_start = max(0, match.start() - 50)
            context_end = min(len(self.text), match.end() + 50)
            context = self.text[context_start:context_end]
            growth_data.append({
                'value': match.group(1),
                'context': context
            })
        
        if growth_data:
            financial_data['growth_mentions'] = growth_data
        
        # Extract CAGR mentions
        cagr_pattern = r'(?i)CAGR[ \t]+of[ \t]+([0-9,.]+)%'
        cagr_matches = re.finditer(cagr_pattern, self.text)
        cagr_data = []
        for match in cagr_matches:
            # Try to get context
            context_start = max(0, match.start() - 50)
            context_end = min(len(self.text), match.end() + 50)
            context = self.text[context_start:context_end]
            cagr_data.append({
                'value': match.group(1),
                'context': context
            })
        
        if cagr_data:
            financial_data['cagr_mentions'] = cagr_data
        
        self.financial_data = financial_data
        return financial_data
    
    def extract_sentiment(self):
        """Extract sentiment from the text"""
        if not self.text:
            self.extract_text()
            
        # Use TextBlob for sentiment analysis
        blob = TextBlob(self.text)
        sentiment = blob.sentiment
        
        # Extract key positive and negative phrases
        sentences = blob.sentences
        positive_phrases = []
        negative_phrases = []
        
        for sentence in sentences:
            if sentence.sentiment.polarity > 0.3:
                positive_phrases.append(str(sentence))
            elif sentence.sentiment.polarity < -0.3:
                negative_phrases.append(str(sentence))
        
        return {
            'polarity': sentiment.polarity,
            'subjectivity': sentiment.subjectivity,
            'sentiment': 'positive' if sentiment.polarity > 0.1 else 'negative' if sentiment.polarity < -0.1 else 'neutral',
            'positive_phrases': positive_phrases[:5],  # Limit to top 5
            'negative_phrases': negative_phrases[:5]   # Limit to top 5
        }
    
    def process(self):
        """Run the full extraction process"""
        self.extract_text()
        self.extract_tables()
        self.identify_sections()
        self.extract_metrics()
        self.extract_financial_data()
        sentiment = self.extract_sentiment()
        
        return {
            'metrics': self.metrics,
            'sections': self.sections,
            'tables': self.tables,
            'financial_data': self.financial_data,
            'sentiment': sentiment,
            'text': self.text
        }

def get_stock_news(ticker):
    """Get latest news for a stock ticker"""
    try:
        # Using Alpha Vantage API for news
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
            
            # Display sector if available
            if 'sector' in metrics:
                st.markdown(f"<div class='metric-card'><b>Sector:</b> {metrics['sector']}</div>", unsafe_allow_html=True)
            
            # Display P/E ratio if available
            if 'pe_ratio' in metrics:
                st.markdown(f"<div class='metric-card'><b>P/E Ratio:</b> {metrics['pe_ratio']}</div>", unsafe_allow_html=True)
    
    # If we have multiple reports, show a consolidated view
    if len(results) > 1:
        st.subheader("Consolidated View")
        
        # Extract target prices and recommendations
        target_prices = []
        recommendations = []
        
        for result in results:
            metrics = result['metrics']
            if 'target_price' in metrics:
                try:
                    target_prices.append(float(metrics['target_price']))
                except:
                    pass
            
            if 'recommendation' in metrics:
                recommendations.append(metrics['recommendation'])
        
        # Show average target price if available
        if target_prices:
            avg_tp = sum(target_prices) / len(target_prices)
            max_tp = max(target_prices)
            min_tp = min(target_prices)
            
            st.markdown(f"""
            <div class='metric-card'>
                <b>Average Target Price:</b> {avg_tp:.2f}<br>
                <b>Range:</b> {min_tp:.2f} - {max_tp:.2f}
            </div>
            """, unsafe_allow_html=True)
        
        # Show recommendation distribution if available
        if recommendations:
            rec_counts = {}
            for rec in recommendations:
                rec_counts[rec] = rec_counts.get(rec, 0) + 1
            
            rec_df = pd.DataFrame({
                'Recommendation': list(rec_counts.keys()),
                'Count': list(rec_counts.values())
            })
            
            # Create a horizontal bar chart
            fig = px.bar(rec_df, y='Recommendation', x='Count', orientation='h',
                        color='Recommendation', color_discrete_map={
                            'BUY': 'green', 'OVERWEIGHT': 'lightgreen', 'OUTPERFORM': 'mediumseagreen',
                            'HOLD': 'gold', 'NEUTRAL': 'orange',
                            'SELL': 'red', 'UNDERWEIGHT': 'tomato', 'REDUCE': 'indianred'
                        })
            
            fig.update_layout(title="Recommendation Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display sentiment comparison
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
            tab1, tab2, tab3, tab4 = st.tabs(["Report Sections", "Financial Data", "Tables", "Market Data"])
            
            with tab1:
                display_sections(results)
            
            with tab2:
                display_financial_data(results)
            
            with tab3:
                display_tables(results)
            
            with tab4:
                fetch_stock_info(results)
            
            # Add a download section for full analysis report
            st.header("Export Analysis")
            if st.button("Generate Full Analysis Report"):
                # Create a combined text report
                report_text = f"# Equity Research Analysis Report\n\n"
                report_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                for i, result in enumerate(results):
                    report_text += f"## Report {i+1}: {result['filename']}\n\n"
                    
                    # Add metrics
                    report_text += "### Key Metrics\n\n"
                    for key, value in result['metrics'].items():
                        report_text += f"- **{key.replace('_', ' ').title()}:** {value}\n"
                    
                    # Add sentiment
                    sentiment = result['sentiment']
                    report_text += f"\n### Sentiment Analysis\n\n"
                    report_text += f"- **Overall Sentiment:** {sentiment['sentiment'].title()}\n"
                    report_text += f"- **Polarity:** {sentiment['polarity']:.2f}\n"
                    report_text += f"- **Subjectivity:** {sentiment['subjectivity']:.2f}\n"
                    
                    # Add sections summary
                    report_text += "\n### Key Sections\n\n"
                    for section_name, content in result['sections'].items():
                        short_content = content[:500] + "..." if len(content) > 500 else content
                        report_text += f"#### {section_name.replace('_', ' ').title()}\n\n{short_content}\n\n"
                    
                    report_text += "---\n\n"
                
                # Add combined analysis if multiple reports
                if len(results) > 1:
                    report_text += "## Consolidated Analysis\n\n"
                    # Add content based on your analysis...
                
                # Offer the report for download
                st.download_button(
                    "Download Analysis as Markdown",
                    report_text,
                    "equity_research_analysis.md",
                    "text/markdown"
                )
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
        
        st.markdown("---")
        st.markdown("### 🔮 Special Features for Financial Reports")
        special_cols = st.columns(2)
        
        with special_cols[0]:
            st.markdown("#### Table Extraction")
            st.markdown("• Automatically extract tables from PDFs\n• Identify financial data in tables\n• Export to CSV for further analysis")
        
        with special_cols[1]:
            st.markdown("#### Financial Insights")
            st.markdown("• Identify growth metrics and CAGR\n• Extract revenue and profit figures\n• Find key financial indicators")

if __name__ == "__main__":
    main()


def display_financial_data(results):
    """Display financial data extracted from the reports"""
    if not results:
        return
    
    st.header("Financial Insights")
    
    # Combine financial data from all reports
    financial_insights = []
    
    for i, result in enumerate(results):
        financial_data = result['financial_data']
        if not financial_data:
            continue
        
        # Add growth mentions
        if 'growth_mentions' in financial_data:
            for mention in financial_data['growth_mentions']:
                financial_insights.append({
                    'Report': f"Report {i+1}",
                    'Type': 'Growth',
                    'Value': mention['value'] + '%',
                    'Context': mention['context']
                })
        
        # Add CAGR mentions
        if 'cagr_mentions' in financial_data:
            for mention in financial_data['cagr_mentions']:
                financial_insights.append({
                    'Report': f"Report {i+1}",
                    'Type': 'CAGR',
                    'Value': mention['value'] + '%',
                    'Context': mention['context']
                })
        
        # Add revenue and profit if available
        if 'revenue' in financial_data:
            financial_insights.append({
                'Report': f"Report {i+1}",
                'Type': 'Revenue',
                'Value': financial_data['revenue'],
                'Context': 'Mentioned in report'
            })
        
        if 'net_profit' in financial_data:
            financial_insights.append({
                'Report': f"Report {i+1}",
                'Type': 'Net Profit',
                'Value': financial_data['net_profit'],
                'Context': 'Mentioned in report'
            })
    
    if financial_insights:
        # Create a DataFrame
        df = pd.DataFrame(financial_insights)
        
        # Display as a styled table
        st.dataframe(df.style.apply(lambda x: ['background-color: #f0f8ff' if i % 2 == 0 else '' for i in range(len(x))], axis=0), use_container_width=True)
    else:
        st.info("No specific financial insights were extracted from the reports.")

def display_tables(results):
    """Display tables extracted from the reports"""
    if not results:
        return
    
    all_tables = []
    for i, result in enumerate(results):
        for table in result['tables']:
            all_tables.append({
                'report_index': i,
                'report_name': result['filename'],
                'title': table['title'],
                'page': table['page'],
                'data': table['data']
            })
    
    if not all_tables:
        st.info("No tables were extracted from the reports.")
        return
    
    st.header("Extracted Tables")
    
    # Create a selectbox to choose tables
    table_options = [f"Report {t['report_index']+1}, Page {t['page']}: {t['title']}" for t in all_tables]
    selected_table = st.selectbox("Select a table to view", table_options)
    
    # Get the selected table index
    selected_index = table_options.index(selected_table)
    table = all_tables[selected_index]
    
    # Convert table data to DataFrame
    if table['data'] and len(table['data']) > 0:
        # Use first row as header if it seems appropriate
        first_row = table['data'][0]
        if all(isinstance(cell, str) and cell != '' for cell in first_row):
            df = pd.DataFrame(table['data'][1:], columns=first_row)
        else:
            df = pd.DataFrame(table['data'])
        
        # Display the table
        st.dataframe(
            df.style.apply(lambda x: ['background-color: #f0f8ff' if i % 2 == 0 else '' for i in range(len(x))], axis=0),
            use_container_width=True
        )
        
        # Offer to download the table as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download this table as CSV",
            csv,
            f"table_{table['report_index']}_{table['page']}.csv",
            "text/csv",
            key=f"download_{selected_index}"
        )
    else:
        st.warning("The selected table appears to be empty.")

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
            
            # Display positive and negative phrases
            if sentiment['positive_phrases']:
                st.markdown("### Key Positive Statements")
                for phrase in sentiment['positive_phrases']:
                    st.markdown(f"✓ _{phrase}_")
            
            if sentiment['negative_phrases']:
                st.markdown("### Key Negative Statements")
                for phrase in sentiment['negative_phrases']:
                    st.markdown(f"⚠ _{phrase}_")

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
            tab1, tab2, tab3, tab4 = st.tabs(["Report Sections", "Financial Data", "Tables", "Market Data"])
            
            with tab1:
                display_sections(results)
            
            with tab2:
                display_financial_data(results)
            
            with tab3:
                display_tables(results)
            
            with tab4:
                fetch_stock_info(results)
            
            # Add a download section for full analysis report
            st.header("Export Analysis")
            if st.button("Generate Full Analysis Report"):
                # Create a combined text report
                report_text = f"# Equity Research Analysis Report\n\n"
                report_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                for i, result in enumerate(results):
                    report_text += f"## Report {i+1}: {result['filename']}\n\n"
                    
                    # Add metrics
                    report_text += "### Key Metrics\n\n"
                    for key, value in result['metrics'].items():
                        report_text += f"- **{key.replace('_', ' ').title()}:** {value}\n"
                    
                    # Add sentiment
                    sentiment = result['sentiment']
                    report_text += f"\n### Sentiment Analysis\n\n"
                    report_text += f"- **Overall Sentiment:** {sentiment['sentiment'].title()}\n"
                    report_text += f"- **Polarity:** {sentiment['polarity']:.2f}\n"
                    report_text += f"- **Subjectivity:** {sentiment['subjectivity']:.2f}\n"
                    
                    # Add sections summary
                    report_text += "\n### Key Sections\n\n"
                    for section_name, content in result['sections'].items():
                        short_content = content[:500] + "..." if len(content) > 500 else content
                        report_text += f"#### {section_name.replace('_', ' ').title()}\n\n{short_content}\n\n"
                    
                    report_text += "---\n\n"
                
                # Add combined analysis if multiple reports
                if len(results) > 1:
                    report_text += "## Consolidated Analysis\n\n"
                    # Add content based on your analysis...
                
                # Offer the report for download
                st.download_button(
                    "Download Analysis as Markdown",
                    report_text,
                    "equity_research_analysis.md",
                    "text/markdown"
                )
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
        
        st.markdown("---")
        st.markdown("### 🔮 Special Features for Financial Reports")
        special_cols = st.columns(2)
        
        with special_cols[0]:
            st.markdown("#### Table Extraction")
            st.markdown("• Automatically extract tables from PDFs\n• Identify financial data in tables\n• Export to CSV for further analysis")
        
        with special_cols[1]:
            st.markdown("#### Financial Insights")
            st.markdown("• Identify growth metrics and CAGR\n• Extract revenue and profit figures\n• Find key financial indicators")

if __name__ == "__main__":
    main()
