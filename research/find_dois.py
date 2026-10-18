"""One-off helper: find correct DOIs by title via Crossref bibliographic search."""
import requests


def query(title, rows=3):
    r = requests.get(
        "https://api.crossref.org/works",
        params={
            "query.bibliographic": title,
            "rows": rows,
            "select": "DOI,title,author,container-title,issued,volume,page",
            "mailto": "student-project@example.edu",
        },
        timeout=25,
    )
    for item in r.json()["message"]["items"]:
        authors = ", ".join(x.get("family", "?") for x in item.get("author", [])[:4])
        year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
        title_ = item["title"][0][:70]
        journal = item.get("container-title", [""])[0][:45]
        print(f"  DOI {item['DOI']} | {title_} | {authors} | {journal} "
              f"| {year} v{item.get('volume', '?')} pp {item.get('page', '?')}")


print("precision-recall plot (Saito & Rehmsmeier):")
query("The precision-recall plot is more informative than the ROC plot")
print()
print("Cox 1958 binary sequences:")
query("The regression analysis of binary sequences Cox 1958")
