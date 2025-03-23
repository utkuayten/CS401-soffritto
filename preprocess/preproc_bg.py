import gzip
filename = "/Users/ozgun/DataspellProjects/CS401-soffritto/newdata/4DNFI4BSJRMF.bedGraph.gz"
with gzip.open(filename, 'rt') as f:
    for line in f:
        print(line.strip())