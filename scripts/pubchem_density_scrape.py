import requests
from xml.etree import ElementTree

pubchem = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/"

data = requests.get(pubchem + "935" + "/xml?heading=Density")

root = ElementTree.fromstring(data.content.decode())
nodes = root.findall(".//{http://pubchem.ncbi.nlm.nih.gov/pug_view}String")
for node in nodes:
    print (node.text)
