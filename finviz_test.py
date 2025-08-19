import streamlit as st
from scripts.scrape_finviz import scrape_finviz_snapshot
from scripts.finviz_chat import chat_with_snapshot
import json

st.set_page_config(page_title='Finviz Dashboard + Chat', layout='wide')
st.title('Finviz Dashboard + Chat')

st.markdown('Enter a ticker to fetch the Finviz snapshot table, view a dashboard of indicators, and chat with an assistant that keeps the snapshot in context.')

# ticker input (auto-fetch on change)
ticker = st.text_input('Ticker', value=st.session_state.get('finviz_ticker', 'A'), key='finviz_ticker')

col1, col2 = st.columns([2, 1])

# Automatic fetch when ticker changes or when no snapshot loaded
with col1:
    snapshot = st.session_state.get('snapshot', {})
    last = st.session_state.get('last_finviz_ticker')
    current = st.session_state.get('finviz_ticker')
    if current and current != last:
        with st.spinner(f'Fetching Finviz snapshot for {current}...'):
            try:
                snapshot, ok = scrape_finviz_snapshot(current)
                if ok:
                    st.session_state['snapshot'] = snapshot
                    st.session_state['last_finviz_ticker'] = current
                    st.success('Snapshot fetched successfully')
                else:
                    st.session_state['snapshot'] = {}
                    st.session_state['last_finviz_ticker'] = current
                    st.error('Snapshot table not found or page structure changed.')
            except Exception as e:
                st.session_state['snapshot'] = {}
                st.session_state['last_finviz_ticker'] = current
                st.exception(e)

    snapshot = st.session_state.get('snapshot', {})

    if snapshot:
        st.subheader(f'Dashboard for {ticker}')
        # Show key indicators as key-value pairs and a compact table
        items = list(snapshot.items())
        # show first 10 in two columns
        for i in range(0, min(len(items), 10), 2):
            left = items[i]
            right = items[i + 1] if i + 1 < len(items) else ('', '')
            c1, c2 = st.columns(2)
            c1.metric(left[0], left[1])
            if right[0]:
                c2.metric(right[0], right[1])

        st.markdown('Full snapshot:')
        st.json(snapshot)
    else:
        st.info('No snapshot loaded. Enter a ticker and click Fetch snapshot.')

with col2:
    st.subheader('Chat with the snapshot')
    # Simple chat UI: conversation stored in session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    chat_input = st.text_area('Your question', height=120)

    if st.button('Send'):
        if not chat_input.strip():
            st.warning('Write a question first')
        else:
            # append user message
            st.session_state['chat_history'].append({'role': 'user', 'content': chat_input})
            try:
                snapshot = st.session_state.get('snapshot', {})
                # call helper without passing api_key so it uses configured env var or fallback
                response = chat_with_snapshot(st.session_state['chat_history'], snapshot, ticker)
                st.session_state['chat_history'].append({'role': 'assistant', 'content': response})
            except Exception as e:
                st.exception(e)

    # render chat
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

    if st.button('Clear chat'):
        st.session_state['chat_history'] = []

st.markdown('This is a local test page — do not expose API keys in public.')
