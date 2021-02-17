import re
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_wiki_formula_and_density(link: str):
    page = BeautifulSoup(requests.get(link).text, "html.parser")
    table = page.find(class_="infobox bordered")
    if not table:
        return

    formula, density, mw = None, None, None

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 2:
            continue
        if tds[0].get_text().strip() == "Chemical formula":
            formula = tds[1].get_text().strip()
        if tds[0].get_text().strip() == "Density":
            density = tds[1].get_text().strip()
        if tds[0].get_text().strip() == "Molar mass":
            mw = tds[1].get_text().strip()

    if formula is None or density is None or mw is None:
        return None
    return formula.split("\n")[0].split(";")[0], density.split(",")[0], mw


def convert_density(text: str) -> float:
    kg = "kg" in text
    match = re.search("[0-9\\.]+", text)
    if match is not None:
        density = float(match.group(0))
        if kg:
            density /= 1e3
        return density
    else:
        return None


def convert_mw(text: str) -> float:
    match = re.search("[0-9\\.]+", text)
    if match is not None:
        mw = float(match.group(0))
        return mw
    else:
        return None


# categories = ["Oxides", "Transition_metal_oxides"]
categories = ["Nitrides"]
wiki = "https://en.wikipedia.org"

links = []
for category in categories:
    soup = BeautifulSoup(
        requests.get(wiki + "/wiki/Category:" + category).text, "html.parser"
    )
    body = soup.find(id="mw-pages")
    for a in body.find_all("a"):
        links.append(wiki + a.get("href"))

results = []

with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(get_wiki_formula_and_density, link): link for link in links
    }

    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            print(futures[future], "success", flush=True)
            if any(x in result[1].lower() for x in ["liquid", "/ml", "/l"]):
                print("Suspected liquid", result[0])
            else:
                results.append(result)
        else:
            print(futures[future], "failed", flush=True)


results = sorted(results, key=lambda x: x[0])

with open("wikinitrides.csv", "w") as fp:
    for result in results:
        formula = re.sub("\\[\\d\\]", "", result[0])
        density = convert_density(result[1])
        mw = convert_mw(result[2])
        if density is not None and mw is not None:
            fp.write(f"{formula},{density},{mw}\n")
