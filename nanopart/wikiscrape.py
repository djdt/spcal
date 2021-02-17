import requests
from bs4 import BeautifulSoup


def get_wiki_formula_and_density(link: str):
    page = BeautifulSoup(requests.get(link).text, "lxml")
    table = page.find(class_="infobox bordered")
    if not table:
        return

    formula = None
    density = None

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 2:
            continue
        if tds[0].get_text().strip() == "Chemical formula":
            formula = tds[1].get_text().strip()
        if tds[0].get_text().strip() == "Density":
            density = tds[1].get_text().strip()

    if formula is None or density is None:
        return None
    return formula, density


transoxides = "https://en.wikipedia.org/wiki/Category:Transition_metal_oxides"

soup = BeautifulSoup(requests.get(transoxides).text, "lxml")

fp = open("wikiscrape.csv", "w")

body = soup.find(id="mw-pages")
for a in body.find_all("a"):
    link = "https://en.wikipedia.org" + a.get("href")
    result = get_wiki_formula_and_density(link)
    if result is not None:
        fp.write(f"{result[0]},{result[1]}\n")

fp.close()
