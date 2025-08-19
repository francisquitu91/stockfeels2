import requests
from bs4 import BeautifulSoup
import json

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/115.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9'
}


def scrape_finviz_snapshot(ticker: str, headers: dict = None):
    """Scrape the Finviz 'snapshot' fundamentals table for a given ticker.

    Returns a dict of key -> value pairs extracted from the snapshot table.
    """
    if headers is None:
        headers = DEFAULT_HEADERS
    url = f"https://finviz.com/quote.ashx?t={ticker}&ty=c&p=d&b=1"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    table = soup.find('table', class_='snapshot-table2')
    if not table:
        # Some pages use different structure; try searching for first table with many tds
        candidates = soup.find_all('table')
        for c in candidates:
            if len(c.find_all('td')) > 20:
                table = c
                break

    if not table:
        return {}, False

    cells = [td.get_text(separator=' ', strip=True) for td in table.find_all('td')]
    data = {}
    # Finviz snapshot table often alternates key / value pairs
    for i in range(0, len(cells) - 1, 2):
        key = cells[i]
        val = cells[i + 1]
        # Normalize key (remove trailing colons, etc.)
        key = key.rstrip(':')
        data[key] = val

    return data, True


def main():
    import sys
    ticker = 'A'
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    try:
        data, ok = scrape_finviz_snapshot(ticker)
        if not ok:
            print(json.dumps({'error': 'snapshot table not found'}, indent=2))
            return
        print(json.dumps({'ticker': ticker, 'snapshot': data}, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({'error': str(e)}))


if __name__ == '__main__':
    main()
