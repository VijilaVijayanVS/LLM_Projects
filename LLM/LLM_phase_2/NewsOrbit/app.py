import streamlit as st
import requests
from datetime import datetime, timedelta
import time
from typing import List, Dict
import hashlib
import json
from bs4 import BeautifulSoup
import re

# Page configuration with pastel theme
st.set_page_config(
    page_title="News Orbit",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced pastel UI with readable text
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    
    .main-header {
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: #1a1a1a;
        margin: 0;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.5);
    }
    
    .main-header p {
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .news-card {
        background: linear-gradient(120deg, #fff9e6 0%, #ffe4b3 50%, #ffd9a3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s;
    }
    
    .news-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    .news-card h3 {
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    .news-card p {
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .news-card strong {
        color: #000;
    }
    
    .news-card small {
        color: #555;
    }
    
    .summary-box {
        background: linear-gradient(135deg, #e8d5ff 0%, #b8e3ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        color: #1a1a1a;
    }
    
    .summary-box strong {
        color: #000;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .metric-card h2 {
        color: #1a1a1a;
    }
    
    .metric-card h3 {
        color: #2c3e50;
    }
    
    .metric-card p {
        color: #555;
    }
    
    .stButton>button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .topic-pill {
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.3rem;
        font-weight: 500;
        color: #1a1a1a;
    }
    
    .loading-text {
        text-align: center;
        font-size: 1.1rem;
        color: #667eea;
        padding: 1rem;
    }
    
    .orbit-logo {
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .orbit-icon {
        font-size: 4rem;
        animation: orbit 3s linear infinite;
    }
    
    @keyframes orbit {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Make all text readable */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #1a1a1a !important;
    }
    
    div[data-testid="stMarkdownContainer"] {
        color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for caching
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'cache_timestamps' not in st.session_state:
    st.session_state.cache_timestamps = {}

# Predefined topics
TRENDING_TOPICS = [
    "Artificial Intelligence",
    "Climate Change",
    "Technology",
    "Healthcare",
    "Space Exploration",
    "Cryptocurrency",
    "Politics",
    "Economy",
    "Sports",
    "Entertainment"
]

def get_cache_key(query: str) -> str:
    """Generate cache key from query"""
    return hashlib.md5(query.lower().encode()).hexdigest()

def is_cache_valid(cache_key: str, max_age_minutes: int = 30) -> bool:
    """Check if cached data is still valid"""
    if cache_key not in st.session_state.cache_timestamps:
        return False
    
    cached_time = st.session_state.cache_timestamps[cache_key]
    current_time = datetime.now()
    age = (current_time - cached_time).total_seconds() / 60
    
    return age < max_age_minutes

@st.cache_data(ttl=1800)
def scrape_google_news(topic: str, max_results: int = 10) -> List[Dict]:
    """Scrape Google News RSS feed for the topic"""
    articles = []
    
    try:
        # Google News RSS feed URL
        search_query = topic.replace(' ', '+')
        rss_url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(rss_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')[:max_results]
            
            for item in items:
                title = item.find('title').text if item.find('title') else 'No title'
                link = item.find('link').text if item.find('link') else ''
                pub_date = item.find('pubDate').text if item.find('pubDate') else ''
                description = item.find('description').text if item.find('description') else 'No description'
                
                # Extract source from title (usually in format "Title - Source")
                source = 'Google News'
                if ' - ' in title:
                    parts = title.split(' - ')
                    if len(parts) > 1:
                        source = parts[-1]
                        title = ' - '.join(parts[:-1])
                
                articles.append({
                    'title': title,
                    'description': description,
                    'url': link,
                    'source': source,
                    'published_at': pub_date,
                    'content': description
                })
        
        return articles
    
    except Exception as e:
        st.warning(f"Error scraping Google News: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def summarize_with_huggingface(articles: List[Dict], topic: str) -> str:
    """Summarize articles using Hugging Face Inference API"""
    
    if not articles:
        return f"No news articles found for '{topic}'."
    
    try:
        # Combine article texts for summarization
        combined_text = f"Topic: {topic}\n\n"
        for i, article in enumerate(articles[:5], 1):  # Use top 5 articles
            combined_text += f"{i}. {article['title']}: {article['description']}\n"
        
        # Limit text length for the model
        if len(combined_text) > 1024:
            combined_text = combined_text[:1024]
        
        # Use Hugging Face Inference API (free tier)
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        
        # No API key needed for public models (rate-limited but free)
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "inputs": combined_text,
            "parameters": {
                "max_length": 200,
                "min_length": 50,
                "do_sample": False
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                summary_text = result[0].get('summary_text', '')
                
                # Format the summary nicely
                formatted_summary = f"""
📊 **AI-Generated Summary for: {topic}**

{summary_text}

📈 **Statistics:**
• Total articles analyzed: {len(articles)}
• Sources: {', '.join(list(set([a['source'] for a in articles[:5]])))}
• Latest update: {articles[0]['published_at'][:25] if articles[0].get('published_at') else 'Recent'}

🔍 **Key Headlines:**
"""
                for i, article in enumerate(articles[:3], 1):
                    formatted_summary += f"\n{i}. {article['title']}"
                
                return formatted_summary
        
        # Fallback to simple summary if API fails
        return create_simple_summary(articles, topic)
        
    except Exception as e:
        st.warning(f"Using fallback summarization. HuggingFace API: {str(e)}")
        return create_simple_summary(articles, topic)

def create_simple_summary(articles: List[Dict], topic: str) -> str:
    """Create a simple summary without external API"""
    if not articles:
        return f"No news articles found for '{topic}'."
    
    total_articles = len(articles)
    sources = list(set([a['source'] for a in articles]))
    
    try:
        latest_time = articles[0]['published_at'][:25] if articles[0].get('published_at') else 'Recent'
    except:
        latest_time = "Recent updates available"
    
    summary = f"""
📊 **News Summary for: {topic}**

📈 **Overview:**
Found {total_articles} recent articles from {len(sources)} news sources covering {topic}.

🔍 **Key Information:**
• Latest update: {latest_time}
• Primary sources: {', '.join(sources[:4])}{'...' if len(sources) > 4 else ''}
• Coverage includes multiple perspectives and breaking developments

📰 **Top Headlines:**
"""
    
    for i, article in enumerate(articles[:5], 1):
        summary += f"\n{i}. {article['title']}"
    
    summary += f"\n\n💡 **Insight:** The topic '{topic}' is actively covered across major news outlets with ongoing developments."
    
    return summary

def display_news_card(article: Dict, index: int):
    """Display individual news article in a card"""
    # Clean description from HTML tags
    description = BeautifulSoup(article['description'], 'html.parser').get_text()
    description = description[:200] + '...' if len(description) > 200 else description
    
    # Format published date
    pub_date = article['published_at'][:25] if article.get('published_at') else 'Recent'
    
    st.markdown(f"""
        <div class="news-card">
            <h3>📰 {article['title']}</h3>
            <p><strong>Source:</strong> <span class="topic-pill">{article['source']}</span></p>
            <p>{description}</p>
            <p><small>🕐 {pub_date}</small></p>
        </div>
    """, unsafe_allow_html=True)
    
    if article['url']:
        # Using columns to properly display the link button without key parameter
        st.link_button("Read Full Article 🔗", article['url'])

# Main App
def main():
    # Header with logo
    st.markdown("""
        <div class="orbit-logo">
            <div class="orbit-icon">🌐</div>
        </div>
        <div class="main-header">
            <h1>News Orbit</h1>
            <p style="font-size: 1.2rem; color: #2c3e50;">Orbiting the World of News with AI</p>
            <p style="font-size: 0.9rem; color: #555;">Powered by Google News RSS + Hugging Face AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Topic selection
        search_mode = st.radio(
            "Select Mode:",
            ["🔥 Trending Topics", "🔍 Custom Search"],
            help="Choose between predefined topics or search your own"
        )
        
        if search_mode == "🔥 Trending Topics":
            selected_topic = st.selectbox(
                "Select a Topic:",
                TRENDING_TOPICS,
                help="Choose from popular news categories"
            )
        else:
            selected_topic = st.text_input(
                "Enter Custom Topic:",
                placeholder="e.g., Quantum Computing",
                help="Search for any topic you're interested in"
            )
        
        # Number of articles
        num_articles = st.slider(
            "Number of Articles:",
            min_value=5,
            max_value=20,
            value=10,
            help="More articles = more comprehensive summary"
        )
        
        # Summarization option
        use_ai_summary = st.checkbox(
            "🤖 Use AI Summarization",
            value=True,
            help="Enable Hugging Face AI for advanced summaries"
        )
        
        # Search button
        search_button = st.button("🚀 Fetch News", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Cache Status")
        cache_count = len(st.session_state.cache)
        st.metric("Cached Topics", cache_count)
        
        if st.button("🗑️ Clear Cache"):
            st.session_state.cache = {}
            st.session_state.cache_timestamps = {}
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 padding: 1rem; border-radius: 10px; color: white; font-size: 0.85rem;">
                <strong>💡 Features:</strong><br>
                ✅ Real-time Google News<br>
                ✅ AI Summarization<br>
                ✅ Smart Caching<br>
                ✅ No API Key Required
            </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if search_button and selected_topic:
        cache_key = get_cache_key(selected_topic)
        
        # Check cache for articles
        if is_cache_valid(cache_key):
            st.success("⚡ Loading from cache (faster!)")
            articles = st.session_state.cache[cache_key]
        else:
            with st.spinner(f"🔎 Scraping Google News for '{selected_topic}'..."):
                articles = scrape_google_news(selected_topic, num_articles)
                
                if articles:
                    # Store in cache
                    st.session_state.cache[cache_key] = articles
                    st.session_state.cache_timestamps[cache_key] = datetime.now()
                    st.success(f"✅ Found {len(articles)} articles!")
                else:
                    st.error("❌ No articles found. Try a different topic or check your connection.")
        
        if articles:
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                    <div class="metric-card">
                        <h2 style="margin:0;">📰</h2>
                        <h3>{}</h3>
                        <p>Articles Found</p>
                    </div>
                """.format(len(articles)), unsafe_allow_html=True)
            
            with col2:
                sources = len(set([a['source'] for a in articles]))
                st.markdown("""
                    <div class="metric-card">
                        <h2 style="margin:0;">📡</h2>
                        <h3>{}</h3>
                        <p>News Sources</p>
                    </div>
                """.format(sources), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div class="metric-card">
                        <h2 style="margin:0;">⚡</h2>
                        <h3>Live</h3>
                        <p>Real-time Data</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Generate and display summary
            st.markdown("## 📝 News Summary")
            
            if use_ai_summary:
                with st.spinner("🤖 Generating AI summary with Hugging Face..."):
                    summary = summarize_with_huggingface(articles, selected_topic)
            else:
                summary = create_simple_summary(articles, selected_topic)
            
            st.markdown(f"""
                <div class="summary-box">
                    {summary.replace(chr(10), '<br>')}
                </div>
            """, unsafe_allow_html=True)
            
            # Display individual articles
            st.markdown("## 📑 Detailed Articles")
            
            # Tabs for better organization
            tab1, tab2 = st.tabs(["📰 All Articles", "📊 Article Analysis"])
            
            with tab1:
                for idx, article in enumerate(articles):
                    with st.container():
                        display_news_card(article, idx)
                        st.markdown("<br>", unsafe_allow_html=True)
            
            with tab2:
                st.markdown("### 📊 Source Distribution")
                source_count = {}
                for article in articles:
                    source = article['source']
                    source_count[source] = source_count.get(source, 0) + 1
                
                # Display as styled cards
                for source, count in sorted(source_count.items(), key=lambda x: x[1], reverse=True):
                    st.markdown(f"""
                        <div style="background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
                             padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;">
                            <strong style="color: #1a1a1a;">{source}</strong>: 
                            <span style="color: #2c3e50;">{count} article(s)</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### 📅 Timeline")
                st.info(f"Articles span from latest updates to recent coverage of '{selected_topic}'")
            
            # Export option
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                export_data = {
                    'topic': selected_topic,
                    'timestamp': datetime.now().isoformat(),
                    'summary': summary,
                    'total_articles': len(articles),
                    'articles': articles
                }
                
                st.download_button(
                    label="📥 Export as JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"news_{selected_topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                text_export = f"NEWS ORBIT SUMMARY: {selected_topic}\n"
                text_export += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                text_export += summary + "\n\n"
                text_export += "="*50 + "\n\n"
                
                for i, article in enumerate(articles, 1):
                    text_export += f"{i}. {article['title']}\n"
                    text_export += f"   Source: {article['source']}\n"
                    text_export += f"   URL: {article['url']}\n\n"
                
                st.download_button(
                    label="📄 Export as Text",
                    data=text_export,
                    file_name=f"news_{selected_topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
    elif not selected_topic and search_button:
        st.warning("⚠️ Please enter a topic to search!")
    
    else:
        # Welcome screen
        st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h2 style="color: #1a1a1a;">👋 Welcome to News Orbit!</h2>
                <p style="font-size: 1.2rem; color: #2c3e50;">
                    Select a trending topic or search for a custom topic to get started.
                </p>
                <p style="margin-top: 2rem; font-size: 1.1rem; color: #555;">
                    🔥 Choose from trending topics<br>
                    🔍 Search any custom topic<br>
                    ⚡ Lightning-fast with caching<br>
                    🤖 AI-powered summaries via Hugging Face<br>
                    📰 Real-time Google News scraping<br>
                    🆓 100% Free - No API keys needed
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display sample trending topics
        st.markdown('<h3 style="color: #1a1a1a;">🌟 Popular Topics to Explore</h3>', unsafe_allow_html=True)
        
        cols = st.columns(5)
        for idx, topic in enumerate(TRENDING_TOPICS[:5]):
            with cols[idx]:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         padding: 1rem; border-radius: 10px; text-align: center; color: white;
                         min-height: 80px; display: flex; align-items: center; justify-content: center;">
                        <strong>{topic}</strong>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        cols2 = st.columns(5)
        for idx, topic in enumerate(TRENDING_TOPICS[5:10]):
            with cols2[idx]:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                         padding: 1rem; border-radius: 10px; text-align: center; color: white;
                         min-height: 80px; display: flex; align-items: center; justify-content: center;">
                        <strong>{topic}</strong>
                    </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #555; padding: 1rem;">
            <p style="color: #1a1a1a;"><strong>🚀 News Orbit - Built for SEQATO LLM Portfolio Development Program</strong></p>
            <p style="color: #555;"><small>⚡ Google News RSS • 🤖 Hugging Face BART-Large-CNN • 💾 Smart Caching • 🎨 Enhanced UI</small></p>
            <p style="color: #555;"><small>💡 No API keys required • 100% Free to use</small></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()