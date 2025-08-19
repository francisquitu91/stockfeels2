import firecrawl, sys, traceback
print('firecrawl.__file__=', getattr(firecrawl,'__file__',None))
print('firecrawl.__version__=', getattr(firecrawl,'__version__',None))
print('public members sample:', [m for m in dir(firecrawl) if not m.startswith('_')][:30])
try:
    from firecrawl import FirecrawlApp
    print('FirecrawlApp import: OK')
except Exception as e:
    print('FirecrawlApp import: FAILED', e)
    traceback.print_exc()
