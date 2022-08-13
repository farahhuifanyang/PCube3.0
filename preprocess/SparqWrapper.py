'''
Author: your name
Date: 2021-04-13 16:36:39
LastEditTime: 2021-05-28 09:06:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/preprocess/sparq.py
'''
from SPARQLWrapper import SPARQLWrapper, JSON


class SparqlWrapper:
    def __init__(self) -> None:
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.sparql.setReturnFormat(JSON)

    def runQuery(self, id1, id2):
        res = []
        self.sparql.setQuery("""
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            select distinct ?p ?pLabel {
                wd:""" + id1+""" ?pd wd:""" + id2 + """ .
                ?p wikibase:directClaim ?pd .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "zh". }
            } group by ?p ?pLabel
        """)
        results = self.sparql.query().convert()

        for result in results["results"]["bindings"]:
            res.append(result["pLabel"]["value"])
        return res


if __name__ == "__main__":
    wrapper = SparqlWrapper()
    print(wrapper.runQuery("Q865", "Q148"))
