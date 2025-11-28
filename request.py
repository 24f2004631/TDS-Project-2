# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx"
# ]
# ///

import httpx

URL1 = "http://localhost:8000/p2"
URL2 = "https://tds-llm-analysis.s-anand.net/submit"
URL3 = "https://24f2004631-tds-project-2.hf.space/p2"


payload = '''
{
  "email": "24f2004631@ds.study.iitm.ac.in",
  "secret": "discruter",
  "url": "https://tds-llm-analysis.s-anand.net/demo",
  "msg": "hello"
}'''

payload2 = """
{
  "email": "24f2004631@ds.study.iitm.ac.in",
  "secret": "discruter",
  "url": "https://tds-llm-analysis.s-anand.net/demo",
  "answer": "hello"
}
"""
payload3 = """
{
  "email": "24f2004631@ds.study.iitm.ac.in",
  "secret": "discruter",
  "url": "https://tds-llm-analysis.s-anand.net/demo-scrape?email=24f2004631%40ds.study.iitm.ac.in&id=20276",
  "answer": "hello"
}
"""

response = httpx.post(URL3, data=payload)
print(response.status_code)
print(response.text)