import sys
from os import listdir
from os.path import isfile, join
import json
import re
import numpy as np
from scipy import sparse
import code
from sklearn.ensamble import RandomForestClassifier

data_path = "data"
rgx = re.compile("\w+")

def main(argv):
  indices = []
  indptr = [0]
  data = []
  vocab = {}
  for file_path in [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]:
    #print(file_path)
    with open(file_path) as fp:
      json_data = json.load(fp)
      for article in filter(
          lambda article: "topics" in article 
            and len(article["topics"])!=0 
            and "body" in article 
            and len(article["body"]) != 0, json_data):

        body = article["body"]
      
        for word in re.findall(rgx, body):
          word = word.lower()
          index = vocab.setdefault(word, len(vocab))
          indices.append(index)
          data.append(1)
        indptr.append(len(indices))

  count_matrix = sparse.csr_matrix((data, indices, indptr), dtype=int)
  code.interact(local=locals())


    
if __name__ == "__main__":
  main(sys.argv)
