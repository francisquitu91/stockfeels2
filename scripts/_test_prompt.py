from scripts.scrape_finviz import scrape_finviz_snapshot
from scripts.finviz_chat import generate_system_prompt

s, ok = scrape_finviz_snapshot('A')
print('ok=', ok)
print(generate_system_prompt(dict(list(s.items())[:10]), 'A'))
