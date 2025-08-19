l to find only the news, also puts info in news_table by ticker
    html = BeautifulSoup(response, 'html')
    news_table = html.find(id='news-table')
    news_tables[tic