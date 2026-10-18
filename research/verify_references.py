"""One-off helper: verify reference DOIs via the Crossref API (polite pool)."""
import time

import requests

DOIS = [
    ("Romero & Ventura 2010", "10.1109/TSMCC.2010.2053532"),
    ("Saito & Rehmsmeier 2015", "10.1371/journal.pone.0117848"),
    ("Chawla et al. 2002 SMOTE", "10.1613/jair.953"),
    ("He & Garcia 2009", "10.1109/TKDE.2008.239"),
    ("Harris et al. 2020 NumPy", "10.1038/s41586-020-2649-2"),
    ("Cox 1958", "10.1111/j.2517-6161.1958.tb00792.x"),
    ("Hastie et al. 2009 ESL", "10.1007/978-0-387-84858-7"),
]

for name, doi in DOIS:
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{doi}",
            params={"mailto": "student-project@example.edu"},
            timeout=20,
        )
        if r.ok:
            m = r.json()["message"]
            authors = ", ".join(
                a.get("family", "?") for a in m.get("author", [])[:4]
            )
            title = m["title"][0][:80]
            journal = m.get("container-title", [""])[0][:45]
            pages = m.get("page", "?")
            volume = m.get("volume", "?")
            year = m.get("issued", {}).get("date-parts", [["?"]])[0][0]
            print(f"OK  {name}: '{title}' | {authors} | {journal} | "
                  f"vol {volume} pp.{pages} ({year})")
        else:
            print(f"FAIL {name}: HTTP {r.status_code}")
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
    time.sleep(1.2)
