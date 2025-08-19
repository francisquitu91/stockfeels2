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
    Display investment strategy chatbot interface - Modern clean design
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
        
        .welcome-section {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, rgba(16, 163, 127, 0.05), rgba(0, 0, 0, 0.3));
            border-radius: 15px;
            margin: 2rem 0;
            border: 1px solid #333;
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
        }
        
        .input-section {
            background: linear-gradient(135deg, rgba(16, 163, 127, 0.05), rgba(0, 0, 0, 0.5));
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            border: 1px solid #333;
            backdrop-filter: blur(10px);
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

    # Initialize chat history
    if "investment_chat_history" not in st.session_state:
        st.session_state.investment_chat_history = []
    
    # Show welcome section or chat history
    if len(st.session_state.investment_chat_history) == 0:
        # Welcome section with examples
        st.markdown("""
        <div class="welcome-section">
            <h2 style="color: white; margin-bottom: 1rem; font-size: 2rem;">Welcome to Investment Strategy Assistant</h2>
            <p style="color: #ccc; font-size: 1.1rem; margin-bottom: 2rem;">
                Get personalized Finviz screening strategies and put options guidance. 
                AI-powered recommendations tailored to your investment style and risk tolerance.
            </p>
            <p style="color: #10a37f; font-weight: 600;">Choose an example below or ask your own question:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💎 Value Stocks for Puts", use_container_width=True, type="secondary"):
                quick_message = "Help me find undervalued large-cap stocks suitable for cash-secured puts with good fundamentals"
                st.session_state.investment_chat_history.append({"role": "user", "content": quick_message})
                with st.spinner("🤔 Analyzing..."):
                    ai_response = generate_investment_advice(quick_message, "Value Investing, Conservative")
                st.session_state.investment_chat_history.append({"role": "assistant", "content": ai_response})
                st.experimental_rerun()
        
        with col2:
            if st.button("💰 High Dividend Strategy", use_container_width=True, type="secondary"):
                quick_message = "Show me high-dividend stocks perfect for generating income with put strategies"
                st.session_state.investment_chat_history.append({"role": "user", "content": quick_message})
                with st.spinner("🤔 Analyzing..."):
                    ai_response = generate_investment_advice(quick_message, "Dividend Income, Conservative")
                st.session_state.investment_chat_history.append({"role": "assistant", "content": ai_response})
                st.experimental_rerun()
        
        with col3:
            if st.button("⚡ Growth Screening", use_container_width=True, type="secondary"):
                quick_message = "Create a Finviz filter for growth stocks suitable for covered put strategies"
                st.session_state.investment_chat_history.append({"role": "user", "content": quick_message})
                with st.spinner("🤔 Analyzing..."):
                    ai_response = generate_investment_advice(quick_message, "Growth Investing, Moderate")
                st.session_state.investment_chat_history.append({"role": "assistant", "content": ai_response})
                st.experimental_rerun()
    
    else:
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
    
    # Investment profile selection
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
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; border-top: 1px solid #404040; color: #666;">
        <p style="margin: 0; font-size: 0.9rem;">
            💡 AI-powered investment assistant • Finviz screening • Put options guidance
        </p>
    </div>
    """, unsafe_allow_html=True)

# Resto del código original...
# [El resto de las funciones permanecen igual desde test_firecrawl_connection() en adelante]
