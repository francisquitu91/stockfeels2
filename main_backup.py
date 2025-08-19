#importing url opener
from urllib.request import urlopen, Request
#importing web scrapers
from bs4 import BeautifulSoup
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#allows us to manipulate data in table structure
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from datetime import date as date2
import plotly.graph_objects as go
import plotly.express as px
import requests
import time
import re
import json
import os
from openai import OpenAI
from firecrawl import FirecrawlApp

def generate_ai_summary(content_data, ticker, sentiment_score):
    """
    Genera un resumen inteligente usando OpenRouter AI
    """
    try:
        # Configurar cliente OpenRouter
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )
        
        # Preparar el contexto para el AI
        full_content = "\n".join([f"- {item['title']}: {item['content'][:500]}..." for item in content_data[:5]])
        
        prompt = f"""
        Analyze the following news about {ticker} and provide a concise summary:
        
        {full_content}
        
        General sentiment score: {sentiment_score:.2f} (-1 very negative, +1 very positive)
        
        Please provide:
        1. Summary of main topics (maximum 3 points)
        2. Potential impact on stock price
        3. Key factors to monitor
        
        Keep the response in English and concise (maximum 200 words).
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating AI analysis: {str(e)}"

def load_finviz_filters():
    """
    Load Finviz filters from JSON file
    """
    try:
        with open('filtros.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading filters: {str(e)}")
        return {}

def generate_investment_advice(user_message, investment_profile=None):
    """
    Generate investment advice using GPT-4.1-nano with Finviz integration
    """
    try:
        # Configure OpenRouter client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )
        
        # Load Finviz filters
        filters = load_finviz_filters()
        
        # Create context about available filters and strategies
        filters_context = f"""
        Available Finviz filters: {json.dumps(filters, indent=2)}
        
        Investment Profile Context: {investment_profile if investment_profile else 'Not specified'}
        """
        
        prompt = f"""
        You are an expert investment advisor specializing in Finviz stock screening and put options strategies. 

        {filters_context}

        User Question: {user_message}

    Guidelines:
        1. Always ask about investment style first if not provided (value, growth, dividend, passive, active)
        2. Use Finviz filters to create specific stock screening strategies
        3. Generate direct Finviz URLs when appropriate
        4. Focus on cash-secured puts and covered put strategies
        5. Explain each filter and its relevance
        6. Provide step-by-step guidance
    7. Keep responses concise and actionable

        Respond in English with practical, actionable advice.
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating investment advice: {str(e)}"

def show_investment_chatbot():
    """
    Display investment strategy chatbot interface - ChatGPT style with navigation
    """
    # Back button and title (responsive + centered)
    st.markdown(
        """
        <style>
        .page-title { 
            color: white; 
            margin: 0; 
            font-size: clamp(1.25rem, 2.5vw, 2rem);
            text-align: center;
            line-height: 1.2;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_center, col_right = st.columns([1, 8, 1])
    with col_left:
        if st.button("← Back", type="secondary"):
            st.session_state.current_page = "landing"
            try:
                st.experimental_set_query_params(page="landing")
            except Exception:
                pass
            st.experimental_rerun()
    with col_center:
        st.markdown('<h1 class="page-title">🎯 Investment Strategy Assistant</h1>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Custom CSS for modern interface matching Sentiment Analysis
    st.markdown("""
    <style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    .chat-container {
        background: linear-gradient(135deg, rgba(16, 163, 127, 0.1), rgba(0, 0, 0, 0.8));
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid #333;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .user-message {
        background: linear-gradient(135deg, #2d2d2d, #1a1a1a);
        padding: 20px 25px;
        border-radius: 20px;
        margin: 15px 0;
        margin-left: 80px;
        color: white;
        border: 1px solid #404040;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        position: relative;
    }
    
    .user-message::before {
        content: "💬";
        position: absolute;
        left: -25px;
        top: 15px;
        font-size: 1.2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1a1a1a, #0d1117);
        padding: 20px 25px;
        border-radius: 20px;
        margin: 15px 0;
        margin-right: 80px;
        color: #e6e6e6;
        border: 1px solid #333;
        border-left: 4px solid #10a37f;
        box-shadow: 0 4px 15px rgba(16, 163, 127, 0.1);
        position: relative;
    }
    
    .assistant-message::before {
        content: "🤖";
        position: absolute;
        right: -25px;
        top: 15px;
        font-size: 1.2rem;
    }
    
    .welcome-section {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, rgba(16, 163, 127, 0.05), rgba(0, 0, 0, 0.3));
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid #333;
    }
    
    .example-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .example-card {
        background: linear-gradient(135deg, #1a1a1a, #0d1117);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .example-card:hover {
        transform: translateY(-5px);
        border-color: #10a37f;
        box-shadow: 0 8px 25px rgba(16, 163, 127, 0.2);
    }
    
    .input-section {
        background: linear-gradient(135deg, rgba(16, 163, 127, 0.05), rgba(0, 0, 0, 0.5));
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid #333;
        position: sticky;
        bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .profile-section {
        background: linear-gradient(135deg, rgba(16, 163, 127, 0.03), rgba(0, 0, 0, 0.2));
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "investment_chat_history" not in st.session_state:
        st.session_state.investment_chat_history = []
    
    # Chat messages container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.investment_chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong style="color: #10a37f;">You</strong><br><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <strong style="color: #10a37f;">AI Assistant</strong><br><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick action suggestions (only show if chat is empty)
    if len(st.session_state.investment_chat_history) == 0:
        st.markdown("""
        <div style="text-align: center; margin: 40px 0;">
            <h3 style="color: #888; margin-bottom: 20px;">Try these examples:</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("� Value Stocks for Puts", use_container_width=True, type="secondary"):
                quick_message = "Help me find undervalued large-cap stocks suitable for cash-secured puts with good fundamentals"
                st.session_state.investment_chat_history.append({"role": "user", "content": quick_message})
                with st.spinner("🤔 Analyzing..."):
                    ai_response = generate_investment_advice(quick_message, "Value Investing, Conservative")
                st.session_state.investment_chat_history.append({"role": "assistant", "content": ai_response})
                st.experimental_rerun()
        
        with col2:
            if st.button("� High Dividend Strategy", use_container_width=True, type="secondary"):
                quick_message = "Show me high-dividend stocks perfect for generating income with put strategies"
                st.session_state.investment_chat_history.append({"role": "user", "content": quick_message})
                with st.spinner("🤔 Analyzing..."):
                    ai_response = generate_investment_advice(quick_message, "Dividend Income, Conservative")
                st.session_state.investment_chat_history.append({"role": "assistant", "content": ai_response})
                st.experimental_rerun()
        
        with col3:
            if st.button("🎯 Growth Screening", use_container_width=True, type="secondary"):
                quick_message = "Create a Finviz filter for growth stocks suitable for covered put strategies"
                st.session_state.investment_chat_history.append({"role": "user", "content": quick_message})
                with st.spinner("🤔 Analyzing..."):
                    ai_response = generate_investment_advice(quick_message, "Growth Investing, Moderate")
                st.session_state.investment_chat_history.append({"role": "assistant", "content": ai_response})
                st.experimental_rerun()
    
    # Input container
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # User input
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_area(
            "",
            placeholder="💬 Ask about investment strategies, Finviz screening, or put options...",
            height=80,
            label_visibility="collapsed",
            key="user_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer
        send_button = st.button("🚀 Send", use_container_width=True, type="primary")
        clear_button = st.button("🗑️ Clear", use_container_width=True, type="secondary")
    
    # Handle send button
    if send_button and user_input.strip():
        # Add user message
        st.session_state.investment_chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate AI response
        with st.spinner("🤔 Thinking..."):
            ai_response = generate_investment_advice(user_input)
        
        # Add AI response
        st.session_state.investment_chat_history.append({
            "role": "assistant", 
            "content": ai_response
        })
        
        # Clear input and refresh
        st.experimental_rerun()
    
    # Handle clear button
    if clear_button:
        st.session_state.investment_chat_history = []
        st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Investment profile selection (enhanced)
    st.markdown('<div class="profile-section">', unsafe_allow_html=True)
    with st.expander("⚙️ Configure Investment Profile", expanded=False):
        st.markdown("""
        <p style="color: #ccc; margin-bottom: 1rem;">
        Configure your investment preferences for more personalized recommendations:
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            investment_style = st.selectbox(
                "Investment Style:",
                ["", "Value Investing", "Growth Investing", "Dividend Income", "Passive Investing", "Active Trading"],
                help="Choose your preferred investment approach"
            )
        
        with col2:
            risk_tolerance = st.selectbox(
                "Risk Tolerance:",
                ["", "Conservative", "Moderate", "Aggressive"],
                help="Select your risk comfort level"
            )
        
        if investment_style or risk_tolerance:
            st.success(f"✅ Profile: {investment_style} | Risk: {risk_tolerance}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; border-top: 1px solid #404040; color: #666;">
        <p style="margin: 0; font-size: 0.9rem;">
            💡 AI-powered investment assistant • Finviz screening • Put options guidance
        </p>
    </div>
    """, unsafe_allow_html=True)

def test_firecrawl_connection():
    """
    Prueba la conexión con Firecrawl
    """
    try:
        app = FirecrawlApp(api_key="fc-6d4cb3a2546c47c38f51a664e19c9216")
        
        # Probar con una URL simple usando la sintaxis correcta
        test_result = app.scrape_url("https://example.com")
        
        if test_result:
            return True, "Firecrawl connection successful"
        else:
            return False, "Firecrawl returned empty result"
            
    except Exception as e:
        return False, f"Firecrawl connection failed: {str(e)}"

def extract_content_with_firecrawl(url):
    """
    Extrae contenido usando Firecrawl como scraper alternativo
    """
    try:
        # Inicializar Firecrawl con la API key
        app = FirecrawlApp(api_key="fc-6d4cb3a2546c47c38f51a664e19c9216")
        
        # Usar la sintaxis correcta de Firecrawl sin parámetros params
        scrape_result = app.scrape_url(url)
        
        if scrape_result:
            # Intentar obtener contenido de diferentes campos posibles
            content = None
            
            # Probar diferentes campos que puede retornar Firecrawl
            if hasattr(scrape_result, 'markdown') and scrape_result.markdown:
                content = scrape_result.markdown
            elif hasattr(scrape_result, 'content') and scrape_result.content:
                content = scrape_result.content
            elif hasattr(scrape_result, 'text') and scrape_result.text:
                content = scrape_result.text
            elif isinstance(scrape_result, dict):
                # Si es un diccionario, intentar diferentes claves
                content = (scrape_result.get('markdown') or 
                          scrape_result.get('content') or 
                          scrape_result.get('text') or
                          scrape_result.get('data', {}).get('markdown') or
                          scrape_result.get('data', {}).get('content'))
            elif isinstance(scrape_result, str):
                content = scrape_result
            
            if content and len(str(content)) > 50:
                # Limpiar el contenido
                content = str(content)
                content = re.sub(r'[#*`\[\]_~]', '', content)  # Remover markdown syntax
                content = re.sub(r'\s+', ' ', content).strip()   # Normalizar espacios
                content = content.replace('\n', ' ')  # Remover saltos de línea
                
                if len(content) > 100:
                    return content[:3000], True  # Limitar a 3000 caracteres
                else:
                    return "Content too short via Firecrawl", False
            else:
                return f"No content found via Firecrawl. Result type: {type(scrape_result)}", False
        else:
            return "No response from Firecrawl", False
            
    except Exception as e:
        return f"Firecrawl error: {str(e)}", False

def extract_article_content_auto(url):
    """
    Extrae contenido usando BeautifulSoup primero, Firecrawl como fallback
    Retorna: (content, success, method_used)
    """
    # Intentar primero con BeautifulSoup (más rápido)
    content, success = extract_content_with_beautifulsoup(url)
    
    if success:
        return content, True, "BeautifulSoup"
    
    # Si BeautifulSoup falla, intentar con Firecrawl silenciosamente
    content_fc, success_fc = extract_content_with_firecrawl(url)
    
    return content_fc, success_fc, "Firecrawl"

def extract_article_content(url, method="auto"):
    """
    Extrae el contenido completo de un artículo usando el método seleccionado
    method: "firecrawl" o "beautifulsoup"
    """
    if method == "firecrawl":
        # Usar Firecrawl
        return extract_content_with_firecrawl(url)
    elif method == "beautifulsoup":
        # Usar BeautifulSoup
        return extract_content_with_beautifulsoup(url)
    else:
        # Default a Firecrawl
        return extract_content_with_firecrawl(url)

def extract_content_with_beautifulsoup(url):
    """
    Extrae contenido usando BeautifulSoup (método clásico)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    alternative_headers = [
        {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        },
        {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Usar headers alternativos en intentos posteriores
            current_headers = headers if attempt == 0 else alternative_headers[attempt % len(alternative_headers)]
            
            response = requests.get(url, headers=current_headers, timeout=10)
            
            if response.status_code == 403:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return "Access denied (403). Using title-only analysis.", False
            
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Detectar si es un press release pagado
            page_text = soup.get_text().lower()
            if 'paid press release' in page_text or 'sponsored content' in page_text:
                return "This is a paid press release. Using title-only analysis.", False
            
            # Remover elementos no deseados
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            
            # Buscar contenido principal
            content_selectors = [
                'article', '.article-content', '.post-content', '.story-body',
                '.content', '.entry-content', '.article-body', 'main',
                '[class*="content"]', '[class*="article"]', '[class*="story"]'
            ]
            
            article_content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        text = element.get_text().strip()
                        if len(text) > len(article_content):
                            article_content = text
                    if len(article_content) > 200:
                        break
            
            # Si no encuentra contenido específico, usar todo el texto
            if len(article_content) < 200:
                article_content = soup.get_text()
            
            # Limpiar el contenido
            article_content = re.sub(r'\s+', ' ', article_content).strip()
            
            # Verificar si el contenido es sustancial
            if len(article_content) > 200:
                return article_content[:2000], True  # Limitar a 2000 caracteres
            else:
                return "Content too short. Using title-only analysis.", False
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return f"Error loading article: {str(e)}", False
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                return f"Error processing article: {str(e)}", False
    
    return "Failed to extract content after all retries.", False

def sentimentAnalysis(tickers):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    for ticker in tickers:
        try:
            url = finviz_url + ticker
            req = Request(url=url, headers={
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response = urlopen(req)
            html = BeautifulSoup(response, 'html.parser')
            news_table = html.find(id='news-table')
            
            if news_table:
                news_tables[ticker] = news_table
            else:
                st.warning(f"⚠️ No news found for {ticker}. This ticker might not exist or have recent news.")
                continue
                
        except Exception as e:
            st.error(f"❌ Error fetching data for {ticker}: {str(e)}")
            continue

    if not news_tables:
        st.error("❌ No valid tickers found or no news available for the provided symbols.")
        return pd.DataFrame()

    parsed_data = []

    for ticker, news_table in news_tables.items():
        for row in news_table.findAll('tr'):
            if row.find('a'):
                title = row.a.text
                link = 'https://finviz.com' + row.a['href'] if row.a.get('href', '').startswith('/') else row.a.get('href', '')
                
                date_data = row.td.text.replace("\r\n", "").split(' ')
                
                if len(date_data) == 21:
                    article_time = date_data[12]
                    date = "Today"
                else:
                    date = date_data[12]
                    article_time = date_data[13] if len(date_data) > 13 else "N/A"
                
                parsed_data.append([ticker, date, article_time, title, link])

    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title', 'link'])
    
    # Extraer contenido completo de los artículos
    full_contents = []
    extraction_success = []
    methods_used = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        status_text.text(f'Extracting content from article {idx + 1}/{len(df)}...')
        progress_bar.progress((idx + 1) / len(df))
        
        if row['link'] and row['link'] != '':
            # Usar método automático: BeautifulSoup → Firecrawl
            content, success, method = extract_article_content_auto(row['link'])
            
            full_contents.append(content)
            extraction_success.append(success)
            methods_used.append(method)
        else:
            full_contents.append("No link available")
            extraction_success.append(False)
            methods_used.append("N/A")
        
        time.sleep(0.1)  # Pequeña pausa para evitar sobrecargar el servidor
    
    df['full_content'] = full_contents
    df['extraction_success'] = extraction_success
    df['method_used'] = methods_used
    
    progress_bar.empty()
    status_text.empty()
    
    # SECCIÓN: ESTADÍSTICAS DE EXTRACCIÓN
    st.markdown('<div class="full-width-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔍 Extraction Status</h2>', unsafe_allow_html=True)
    
    success_count = sum(extraction_success)
    method_stats = {}
    for method in methods_used:
        method_stats[method] = method_stats.get(method, 0) + 1
    
    # Métricas principales en columnas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Articles Processed", 
            value=len(df),
            delta="Total found"
        )
    with col2:
        success_rate = (success_count / len(df)) * 100 if len(df) > 0 else 0
        st.metric(
            label="Success Rate", 
            value=f"{success_rate:.1f}%",
            delta=f"{success_count}/{len(df)} successful"
        )
    with col3:
        primary_method = max(method_stats.items(), key=lambda x: x[1])[0] if method_stats else "N/A"
        st.metric(
            label="Primary Method",
            value=primary_method,
            delta="Most used"
        )
    
    # Detalles de métodos utilizados
    if method_stats:
        method_details = []
        for method, count in method_stats.items():
            if method != "N/A":
                percentage = (count / len(df)) * 100
                method_details.append(f"**{method}**: {count} articles ({percentage:.1f}%)")
        
        if method_details:
            st.info("**Scraping methods distribution:** " + " | ".join(method_details))
    
    if success_count == 0:
        st.error("❌ Could not extract content from any article. Please verify the URLs.")
        return
    
    # Separador
    st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)

    # Análisis de sentimiento
    vader = SentimentIntensityAnalyzer()

    def analyze_sentiment(row):
        # Si la extracción fue exitosa, analizar el contenido completo
        if row['extraction_success'] and len(row['full_content']) > 50:
            text_to_analyze = f"{row['title']} {row['full_content']}"
        else:
            # Fallback al título solamente
            text_to_analyze = row['title']
        
        return vader.polarity_scores(text_to_analyze)['compound']

    df['compound'] = df.apply(analyze_sentiment, axis=1)
    
    # Procesar fechas
    df.loc[df['date'] == "Today", 'date'] = date2.today()
    df['date'] = pd.to_datetime(df.date).dt.date

    return df

def create_sentiment_gauge(sentiment_score, ticker):
    """
    Crea un gauge chart para mostrar el sentimiento
    """
    # Convertir el score de -1,1 a 0,100 para el gauge
    gauge_value = (sentiment_score + 1) * 50
    
    # Determinar color basado en el sentimiento
    if sentiment_score >= 0.1:
        color = "green"
        sentiment_text = "Positive"
    elif sentiment_score <= -0.1:
        color = "red"
        sentiment_text = "Negative"
    else:
        color = "yellow"
        sentiment_text = "Neutral"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = gauge_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"Sentimiento para {ticker}",
            'font': {'color': 'white', 'size': 16}
        },
        delta = {
            'reference': 50,
            'font': {'color': 'white'}
        },
        gauge = {
            'axis': {
                'range': [None, 100],
                'tickcolor': 'white',
                'tickfont': {'color': 'white'}
            },
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "rgba(255,0,0,0.2)"},
                {'range': [30, 70], 'color': "rgba(255,255,0,0.2)"},
                {'range': [70, 100], 'color': "rgba(0,255,0,0.2)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': 90
            }
        },
        number = {'font': {'color': 'white', 'size': 20}}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    
    return fig, sentiment_text

def show_page():
    # CSS personalizado para modo oscuro y diseño moderno
    st.markdown("""
    <style>
    /* Dark mode background */
    .main > div {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c1427 0%, #12203e 100%);
        color: white !important;
    }
    
    /* Asegurar que todo el texto sea blanco */
    * {
        color: white !important;
    }
    
    /* Específico para elementos de Streamlit */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stText, p, span, div {
        color: white !important;
    }
    
    /* Subheaders específicos */
    .css-10trblm, .css-1629p8f h1, .css-1629p8f h2, .css-1629p8f h3 {
        color: white !important;
    }
    
    /* Header styling */
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    /* Cards with glassmorphism effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
        width: 100%;
        max-width: none;
        box-sizing: border-box;
    }
    
    /* Secciones principales con ancho completo */
    .full-width-section {
        width: 100% !important;
        max-width: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Headers con ancho completo */
    .section-header {
        width: 100% !important;
        text-align: left !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Custom metrics */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="metric-container"] > div {
        color: white !important;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9);
        color: black !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Text input labels */
    .stTextInput > label {
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: white !important;
    }
    
    /* Success/Warning/Error messages */
    .stAlert {
        color: white !important;
    }
    
    /* Logo animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .logo {
        animation: float 3s ease-in-out infinite;
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Remove default streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal con logo
    st.markdown("""
    <div class="main-header">
        <div class="logo">📈</div>
        <h1 style="margin: 0; font-size: 3rem; color: white; font-weight: 700;">
           Stock Sentiment Analyzer
        </h1>
        <p style="font-size: 1.3rem; opacity: 0.9; margin-top: 1rem; color: #00d4ff;">
           Automated sentiment analysis with AI for intelligent investment decisions
        </p>
        <p style="font-size: 1rem; opacity: 0.7; margin-top: 0.5rem;">
           🤖 Smart scraping • Real-time analysis • 🎯 Actionable insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input personalizado para tickers
    st.markdown("""
    <div class="glass-card">
        <h3>🎯 Analysis Configuration</h3>
        <p>Enter the stock symbols you want to analyze, separated by commas</p>
        <p style="font-size: 0.9rem; opacity: 0.7;">Examples: AAPL, MSFT, GOOGL, TSLA, NVDA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input de usuario
    user_input = st.text_input(
        "Stock symbols:",
        value="",
        help="Enter symbols separated by commas",
        label_visibility="collapsed"
    )
    
    # Información sobre el método automático
    st.markdown("""
    <div class="glass-card">
        <h3>🤖 Smart Content Extraction</h3>
        <p>The system automatically uses <strong>BeautifulSoup</strong> for faster processing, and <strong>Firecrawl</strong> as fallback for higher precision when needed.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Procesar input
    if user_input:
        # Limpiar y validar tickers
        raw_tickers = [ticker.strip().upper() for ticker in user_input.split(',') if ticker.strip()]
        
        # Filtrar solo símbolos válidos (letras únicamente, 1-5 caracteres)
        tickers = []
        invalid_tickers = []
        
        for ticker in raw_tickers:
            # Remover caracteres no alfabéticos
            clean_ticker = ''.join(c for c in ticker if c.isalpha())
            
            if len(clean_ticker) >= 1 and len(clean_ticker) <= 5:
                tickers.append(clean_ticker)
            else:
                invalid_tickers.append(ticker)
        
        # Mostrar advertencias para tickers inválidos
        if invalid_tickers:
            st.warning(f"⚠️ Invalid symbols ignored: {', '.join(invalid_tickers)}")
        
        if not tickers:
            st.error("❌ No valid symbols found. Please enter valid stock symbols (e.g., AAPL, MSFT)")
            return
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Analysis", use_container_width=True):
                if tickers:
                    with st.spinner('Analyzing news and sentiment...'):
                        df = sentimentAnalysis(tickers)
                    
                    # Verificar si obtuvimos datos
                    if df.empty:
                        st.error("❌ Could not obtain data for any of the provided symbols. Please verify that the symbols are valid.")
                        return
                    
                    # Mostrar resultados
                    st.markdown("""
                    <div class="glass-card">
                        <h2>Resultados del Análisis</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Métricas generales
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Positive News", len(df[df['compound'] > 0.1]))
                    with col2:
                        st.metric("Negative News", len(df[df['compound'] < -0.1]))
                    with col3:
                        st.metric("Neutral News", len(df[(df['compound'] >= -0.1) & (df['compound'] <= 0.1)]))
                    with col4:
                        st.metric("Total Articles", len(df))
                    
                    # Información sobre la calidad del análisis
                    st.subheader("Content Analysis Quality")
                    
                    # Contar artículos con contenido exitosamente extraído
                    successful_extractions = len(df[df['extraction_success'] == True])
                    failed_extractions = len(df[df['extraction_success'] == False])
                    press_releases = len(df[df['full_content'].str.contains('paid press release', case=False, na=False)])
                    success_rate = (successful_extractions / len(df)) * 100 if len(df) > 0 else 0
                    
                    quality_col1, quality_col2, quality_col3 = st.columns(3)
                    
                    with quality_col1:
                        st.metric(
                            "✅ Successful Extractions", 
                            f"{successful_extractions}/{len(df)}",
                            help="Number of articles where full content was successfully extracted"
                        )
                    
                    with quality_col2:
                        st.metric(
                            "Extraction Success Rate", 
                            f"{success_rate:.1f}%",
                            help="Percentage of articles with successful content extraction"
                        )
                    
                    with quality_col3:
                        st.metric(
                            "📰 Press Releases", 
                            f"{press_releases}",
                            help="Number of paid press releases (analyzed by title only)"
                        )
                    
                    # Mostrar insights sobre la calidad
                    if success_rate > 80:
                        st.success(f"✅ **High Quality Analysis**: {success_rate:.1f}% of articles analyzed with full content")
                    elif success_rate > 60:
                        st.warning(f"⚠️ **Good Analysis**: {success_rate:.1f}% of articles analyzed with full content. {failed_extractions} articles used title-only analysis.")
                    else:
                        st.error(f"⚠️ **Limited Analysis**: Only {success_rate:.1f}% of articles analyzed with full content. {failed_extractions} articles used title-only analysis.")
                    
                    # Análisis por ticker
                    st.markdown('<div class="full-width-section">', unsafe_allow_html=True)
                    st.markdown('<h2 class="section-header">Detailed Analysis by Stock</h2>', unsafe_allow_html=True)
                    
                    # Crear análisis individual para cada ticker
                    for ticker in tickers:
                        ticker_data = df[df['ticker'] == ticker]
                        
                        if len(ticker_data) > 0:
                            avg_sentiment = ticker_data['compound'].mean()
                            
                            # Card para cada acción con mejor estructura
                            st.markdown(f"""
                            <div class="glass-card" style="width: 100%; max-width: none;">
                                <h3 style="margin-bottom: 15px;">🏢 {ticker.upper()}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # GAUGE PRINCIPAL - Ancho completo y muy grande
                            st.markdown("### Sentiment Gauge")
                            fig, sentiment_text = create_sentiment_gauge(avg_sentiment, ticker)
                            # Gauge dominante en toda la pantalla
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                            
                            # MÉTRICAS EN COLUMNAS DEBAJO DEL GAUGE
                            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                            with col_metrics1:
                                st.metric("Articles Analyzed", len(ticker_data))
                            with col_metrics2:
                                st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
                            with col_metrics3:
                                st.metric("Trend", sentiment_text)
                            
                            # AI ANALYSIS - Full width below
                            st.markdown("### 🤖 Artificial Intelligence Analysis")
                            
                            # Prepare data for AI
                            ticker_articles = ticker_data.to_dict('records')
                            content_data = []
                            
                            for article in ticker_articles:
                                content_data.append({
                                    'title': article['title'],
                                    'content': article['full_content'] if article['extraction_success'] else article['title']
                                })
                            
                            # AI analysis in full-width container
                            with st.container():
                                with st.spinner(f'Generating intelligent analysis for {ticker}...'):
                                    ai_summary = generate_ai_summary(content_data, ticker, avg_sentiment)
                                # Display in full-width container
                                st.markdown(f"""
                                <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; border-left: 4px solid #00d4ff; width: 100%; max-width: none; box-sizing: border-box; margin-top: 1rem;">
                                    {ai_summary}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Separator between stocks if not the last one
                            if ticker != tickers[-1]:
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.markdown("---")
                                st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Cerrar sección de análisis detallado
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Separador visual
                    st.markdown("---")
                    
                    # SECCIÓN 2: DETALLES DE ARTÍCULOS (sección de análisis por acción eliminada)
                    st.markdown('<div class="full-width-section">', unsafe_allow_html=True)
                    st.markdown('<h2 class="section-header">📋 Article Details</h2>', unsafe_allow_html=True)
                    
                    # CSS específico para mejorar la tabla
                    st.markdown("""
                    <style>
                    /* Mejorar la visualización de la tabla */
                    .stDataFrame {
                        width: 100% !important;
                    }
                    .stDataFrame > div {
                        width: 100% !important;
                        overflow-x: auto !important;
                        scroll-behavior: smooth !important;
                    }
                    /* Hacer que las celdas de la tabla sean más legibles */
                    .stDataFrame table {
                        width: 100% !important;
                        min-width: 1200px !important;
                    }
                    .stDataFrame th, .stDataFrame td {
                        padding: 8px 12px !important;
                        text-align: left !important;
                        white-space: nowrap !important;
                        overflow: hidden !important;
                        text-overflow: ellipsis !important;
                    }
                    /* Títulos de artículos más largos */
                    .stDataFrame td:nth-child(2) {
                        max-width: 400px !important;
                        white-space: normal !important;
                        word-wrap: break-word !important;
                    }
                    /* URLs más legibles y clicables */
                    .stDataFrame td:nth-child(3) {
                        max-width: 300px !important;
                        white-space: nowrap !important;
                        overflow: hidden !important;
                        text-overflow: ellipsis !important;
                        color: #00d4ff !important;
                        text-decoration: underline !important;
                        cursor: pointer !important;
                    }
                    
                    .stDataFrame td:nth-child(3):hover {
                        color: #ffffff !important;
                        background-color: rgba(0, 212, 255, 0.2) !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Crear tabla más legible y organizada
                    display_df = df[['ticker', 'title', 'link', 'compound', 'extraction_success', 'method_used']].copy()
                    
                    # Acortar títulos muy largos para mejor visualización
                    display_df['title_short'] = display_df['title'].apply(
                        lambda x: x[:80] + "..." if len(x) > 80 else x
                    )
                    
                    # Crear URLs limpias para la tabla
                    display_df['url_clean'] = display_df['link'].apply(
                        lambda x: x if pd.notna(x) and x != '' and str(x).startswith('http') else 'No URL available'
                    )
                    
                    display_df['Sentiment'] = display_df['compound'].apply(
                        lambda x: '📈 Positive' if x > 0.1 else ('📉 Negative' if x < -0.1 else '⚖️ Neutral')
                    )
                    display_df['Extraction'] = display_df['extraction_success'].apply(
                        lambda x: '✅ Full Content' if x else '📰 Title Only'
                    )
                    display_df['Method'] = display_df['method_used'].fillna('N/A')
                    display_df['Score'] = display_df['compound'].round(3)
                    
                    # Rename columns for display
                    display_columns = {
                        'ticker': 'Symbol',
                        'title_short': 'Article Title',
                        'url_clean': 'News URL',
                        'Sentiment': 'Trend',
                        'Score': 'Score',
                        'Extraction': 'Content',
                        'Method': 'Scraper Used'
                    }
                    
                    display_df_final = display_df[['ticker', 'title_short', 'url_clean', 'Sentiment', 'Score', 'Extraction', 'Method']].copy()
                    display_df_final.columns = list(display_columns.values())
                    
                    # Mostrar información sobre la tabla
                    st.info(f"**{len(display_df_final)} articles** analyzed in total.")
                    
                    st.dataframe(
                        display_df_final,
                        use_container_width=True,
                        hide_index=True,
                        height=600,  # Altura fija amplia para mostrar más filas
                        column_config={
                            "Symbol": st.column_config.TextColumn(
                                "Symbol",
                                width="small",
                                help="Stock symbol"
                            ),
                            "Article Title": st.column_config.TextColumn(
                                "Article Title",
                                width="large",
                                help="Complete news article title"
                            ),
                            "News URL": st.column_config.TextColumn(
                                "News URL",
                                width="medium",
                                help="Link to original article"
                            ),
                            "Trend": st.column_config.TextColumn(
                                "Trend",
                                width="small",
                                help="Sentiment classification"
                            ),
                            "Score": st.column_config.NumberColumn(
                                "Score",
                                width="small",
                                help="Numerical sentiment score (-1 to 1)",
                                format="%.3f"
                            ),
                            "Content": st.column_config.TextColumn(
                                "Content",
                                width="medium",
                                help="Type of analysis performed"
                            ),
                            "Scraper Used": st.column_config.TextColumn(
                                "Scraper Used",
                                width="small",
                                help="Extraction method used"
                            )
                        }
                    )
                    
                    # Cerrar última sección
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("⚠️ Please enter at least one valid stock symbol.")

def main():
    """
    Main application with modern landing page
    """
    # Set page configuration
    st.set_page_config(
        page_title="Stock Analysis Suite",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for modern landing page
    st.markdown("""
    <style>
    /* Global background */
    .stApp {
        background-color: #000000 !important;
    }
    
    .main .block-container {
        background-color: #000000 !important;
        padding-top: 0rem;
    }
    
    .main-header {
        text-align: center;
        padding: 60px 0 40px 0;
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 2px solid #10a37f;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 1rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #10a37f;
        margin: 0 0 2rem 0;
        font-weight: 500;
    }
    
    .hero-description {
        font-size: 1.1rem;
        color: #ccc;
        max-width: 600px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #111111 0%, #1a1a1a 100%);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #333;
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        position: relative;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(16, 163, 127, 0.2);
        border-color: #10a37f;
    }
    
    .clickable-card {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 10;
        background: transparent;
        border: none;
        cursor: pointer;
    }
    
    /* Link wrapper for clickable cards */
    a.card-link {
        text-decoration: none;
        color: inherit;
        display: block;
    }
    a.card-link:focus { outline: none; }
    
    /* Make cards clickable with hover effects */
    .clickable-card {
        transition: all 0.3s ease;
        position: relative;
    }
    
    .clickable-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(16, 163, 127, 0.3);
        border-color: #10a37f;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0 0 1rem 0;
    }
    
    .feature-description {
        color: #ccc;
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }
    
    .cta-button {
        background: linear-gradient(135deg, #10a37f 0%, #0d8a66 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 25px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
    }
    
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(16, 163, 127, 0.4);
    }
    
    .stats-container {
        background: #0a0a0a;
        padding: 40px 0;
        margin: 3rem -1rem 2rem -1rem;
        border-top: 1px solid #333;
        border-bottom: 1px solid #333;
    }
    
    .stat-item {
        text-align: center;
        color: white;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #10a37f;
        display: block;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #ccc;
        margin-top: 0.5rem;
    }
    
    .navigation-pills {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .nav-pill {
        background: #1a1a1a;
        color: white;
        padding: 12px 24px;
        border-radius: 25px;
        border: 1px solid #333;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .nav-pill:hover, .nav-pill.active {
        background: #10a37f;
        border-color: #10a37f;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize page state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "landing"
    
    # Sync state with URL query params
    try:
        params = st.experimental_get_query_params()
        if "page" in params:
            page_val = params["page"][0] if isinstance(params["page"], list) else params["page"]
            if page_val in ("landing", "sentiment", "investment"):
                st.session_state.current_page = page_val
    except Exception:
        pass
    
    # Landing Page
    if st.session_state.current_page == "landing":
        show_landing_page()
    elif st.session_state.current_page == "sentiment":
        show_sentiment_analysis()
    elif st.session_state.current_page == "investment":
        show_investment_chatbot()

def show_landing_page():
    """
    Display the main landing page
    """
    # Hero Section with your logo
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 2rem;">
            <img src="https://i.postimg.cc/PrPKRBTq/Foto-COMERCIANDOLA.png" 
                 style="height: 80px; margin-right: 20px; border-radius: 10px;" 
                 alt="ComercianDola Logo">
            <div>
                <h1 class="hero-title" style="margin: 0;">Stock Analysis Suite</h1>
                <p class="hero-subtitle" style="margin: 0;">Professional Financial Analysis Tools</p>
            </div>
        </div>
        <p class="hero-description">
            Advanced AI-powered stock analysis platform combining sentiment analysis, 
            investment strategies, and market intelligence for smarter trading decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section - Clickable Cards
    st.markdown("""
    <div style="max-width: 1200px; margin: 3rem auto; padding: 0 2rem;">
        <h2 style="text-align: center; color: white; font-size: 2.5rem; margin-bottom: 3rem;">
            Choose Your Analysis Tool
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        # Clickable card via URL param
        st.markdown("""
        <a class="card-link" href="?page=sentiment">
            <div class="feature-card clickable-card" style="cursor: pointer;">
                <span class="feature-icon">📈</span>
                <h3 class="feature-title">Sentiment Analysis</h3>
                <p class="feature-description">
                    Analyze market sentiment from news articles using advanced AI. 
                    Get real-time sentiment scores, trends, and intelligent summaries 
                    for any stock symbol to make informed trading decisions.
                </p>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        # Clickable card via URL param
        st.markdown("""
        <a class="card-link" href="?page=investment">
            <div class="feature-card clickable-card" style="cursor: pointer;">
                <span class="feature-icon">🎯</span>
                <h3 class="feature-title">Investment Strategy Assistant</h3>
                <p class="feature-description">
                    Get personalized Finviz screening strategies and put options guidance. 
                    AI-powered chatbot helps you find the perfect stocks for your 
                    investment style and risk tolerance.
                </p>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    # API Logos Carousel Section
    st.markdown("""
    <div style="margin: 4rem 0; padding: 3rem 0; background: linear-gradient(to bottom, rgba(10,10,10,0.5), rgba(0,0,0,1)); border-top: 1px solid #333; border-bottom: 1px solid #333;">
        <div style="max-width: 1200px; margin: 0 auto; text-align: center;">
            <h3 style="color: white; font-size: 2rem; margin-bottom: 2rem; font-weight: 600;">
                APIs that Power Our Services
            </h3>
            <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 3rem; opacity: 0.8;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_Logo.svg" 
                     style="height: 40px; filter: brightness(0.8);" alt="OpenAI">
                <img src="https://www.vectorlogo.zone/logos/twilio/twilio-icon.svg" 
                     style="height: 40px; filter: brightness(0.8);" alt="Twilio">
                <img src="https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg" 
                     style="height: 40px; filter: brightness(0.8);" alt="Google Cloud">
                <img src="https://i.ibb.co/DrqZ7Sc/logo-6813a1b5.png" 
                     style="height: 40px; filter: brightness(0.8);" alt="Vapi">
                <img src="https://i.ibb.co/M6XR8VZ/img-swapcard.png" 
                     style="height: 40px; filter: brightness(0.8);" alt="Deepgram">
            </div>
            <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 3rem; margin-top: 2rem; opacity: 0.8;">
                <img src="https://i.ibb.co/VpqN1Kc/Bolt.jpg" 
                     style="height: 40px; filter: brightness(0.8); border-radius: 8px;" alt="Bolt.new">
                <img src="https://i.ibb.co/92JrX6Z/logo-make.png" 
                     style="height: 40px; filter: brightness(0.8);" alt="Make.com">
                <img src="https://i.ibb.co/8dbgMLG/Eleven-Labs-Logo.webp" 
                     style="height: 40px; filter: brightness(0.8);" alt="Eleven Labs">
                <img src="https://i.ibb.co/T09FTS2/Netlify-logo-2-svg.png" 
                     style="height: 40px; filter: brightness(0.8);" alt="Netlify">
                <img src="https://i.ibb.co/zJmdcF0/hostinger-logo-freelogovectors-net.png" 
                     style="height: 40px; filter: brightness(0.8);" alt="Hostinger">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional Features Section
    st.markdown("""
    <div style="max-width: 1200px; margin: 4rem auto 2rem auto; padding: 0 2rem;">
        <h2 style="text-align: center; color: white; font-size: 2rem; margin-bottom: 2rem;">
            Why Choose Stock Analysis Suite?
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <span style="font-size: 2.5rem; color: #10a37f;">🤖</span>
            <h4 style="color: white; margin: 1rem 0 0.5rem 0;">AI-Powered</h4>
            <p style="color: #ccc; font-size: 0.9rem;">
                Advanced GPT-5 integration for intelligent analysis and recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <span style="font-size: 2.5rem; color: #10a37f;">⚡</span>
            <h4 style="color: white; margin: 1rem 0 0.5rem 0;">Real-Time</h4>
            <p style="color: #ccc; font-size: 0.9rem;">
                Live market data and instant sentiment analysis for quick decisions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <span style="font-size: 2.5rem; color: #10a37f;">🎯</span>
            <h4 style="color: white; margin: 1rem 0 0.5rem 0;">Personalized</h4>
            <p style="color: #ccc; font-size: 0.9rem;">
                Tailored strategies based on your investment style and risk tolerance
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 4rem; padding: 2rem; border-top: 1px solid #404040; color: #666;">
        <p style="margin: 0; font-size: 0.9rem;">
            Built with ❤️ for smarter investing • Powered by AI • Real-time Market Data
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_sentiment_analysis():
    """
    Display sentiment analysis with back navigation
    """
    # Back button and title
    col1, col2 = st.columns([1, 8])
    
    with col1:
        if st.button("← Back", type="secondary"):
            st.session_state.current_page = "landing"
            try:
                st.experimental_set_query_params(page="landing")
            except Exception:
                pass
            st.experimental_rerun()
    
    with col2:
        # Title is handled by show_page() - no duplicate needed here
        pass
    
    st.markdown("---")
    
    # Call the original sentiment analysis function
    show_page()

if __name__ == "__main__":
    main()