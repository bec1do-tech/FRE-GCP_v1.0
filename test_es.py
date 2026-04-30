import os
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:3128"
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:3128"

from elasticsearch import Elasticsearch
from elastic_transport import RequestsHttpNode

es = Elasticsearch(
    "https://my-elasticsearch-project-a8393b.es.us-central1.gcp.elastic.cloud:443",
    api_key="Wl9ndXRKMEIyVFRDLW9UNWV0M2Y6RDFYOWxEZmxEb0RHY24wWWZfa19EZw==",
    node_class=RequestsHttpNode,
    request_timeout=15,
)
print("ping:", es.ping())
print("version:", es.info()["version"]["number"])
print("ES client OK")
