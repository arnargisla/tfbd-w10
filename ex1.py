import sys
import json
import re
import code
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from random import sample
from math import floor
from hashlib import md5

data_path = "data"
rgx = re.compile("\w+")

def main(argv):
  runs = [
      {
        "name": "normal", 
        "word_function": lambda word: word,
      },
      { "name": "hashing_1",
        "word_function": lambda word: int(md5(word.encode("utf-8")).hexdigest(), 16) % 1
      },
      {
        "name": "hashing_50",
        "word_function": lambda word: int(md5(word.encode("utf-8")).hexdigest(), 16) % 50
      },
      {
        "name": "hashing_250",
        "word_function": lambda word: int(md5(word.encode("utf-8")).hexdigest(), 16) % 250
      },
      {
        "name": "hashing_500",
        "word_function": lambda word: int(md5(word.encode("utf-8")).hexdigest(), 16) % 500
      },
      {
        "name": "hashing_1000",
        "word_function": lambda word: int(md5(word.encode("utf-8")).hexdigest(), 16) % 1000
      },
      {
        "name": "hashing_2500",
        "word_function": lambda word: int(md5(word.encode("utf-8")).hexdigest(), 16) % 2500
      },
      {
        "name": "hashing_5000",
        "word_function": lambda word: int(md5(word.encode("utf-8")).hexdigest(), 16) % 5000
      },
      {
        "name": "hashing_10000",
        "word_function": lambda word: int(md5(word.encode("utf-8")).hexdigest(), 16) % 10000
      },
    ]

  print("name, success_rate, success_count")
  for run in runs:
    indices = []
    indptr = [0]
    data = []
    labels = []
    vocab = {}
    for file_path in [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]:
      with open(file_path) as fp:
        json_data = json.load(fp)
        for article in json_data:
          if "topics" not in article: continue
          if "body" not in article: continue
          if len(article["topics"])==0: continue
          if len(article["body"])==0: continue

          words = re.findall(rgx, article["body"])

          if "earn" in article["topics"]:
            labels.append(True)
          else:
            labels.append(False)

          for word in words:
            word = word.lower()
            word = run["word_function"](word)
            index = vocab.setdefault(word, len(vocab))
            indices.append(index)
            data.append(1)
          indptr.append(len(indices))

    labels = np.array(labels)

    success_rate, success_count = classify(data, indices, indptr, labels)
    print("{},\t{:2.5f}%,\t{}".format(run["name"], success_rate * 100, success_count))


sample_indices = None
inverted_indices = None
def classify(data, indices, indptr, y):
  global sample_indices, inverted_indices
  count_matrix = sparse.csr_matrix((data, indices, indptr), dtype=int)
  row_count = count_matrix.shape[0]
  if sample_indices:
    # Use same indices between runs, so the splits are the same for all models
    pass
  else:
    sample_indices = sample(range(row_count), floor(row_count*0.8))
    inverted_indices = [i for i in range(row_count) if i not in sample_indices]

  train_data = count_matrix[sample_indices]
  test_data = count_matrix[inverted_indices]
  train_labels = y[sample_indices]
  test_labels = y[inverted_indices]

  clf = RandomForestClassifier(n_estimators=50)
  clf.fit(train_data, train_labels)

  prediction = clf.predict(test_data)
  success_count = len(list(filter(lambda a:a, (prediction == test_labels))))
  success_rate = success_count * 1.0/test_labels.shape[0]
  return (success_rate, success_count)



if __name__ == "__main__":
  main(sys.argv)




