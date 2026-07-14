import urllib.request\nresp=urllib.request.urlopen(" http://localhost:8501\, timeout=10)\ndata=resp.read(2000)\nprint(resp.status)\nprint(data.decode(\utf-8\, errors=\replace\))
