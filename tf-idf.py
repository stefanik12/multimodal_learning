import math
import sys
from textblob import TextBlob as tb
import numpy as np


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)


def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))


def compute_tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


def load_stopfile(stopword_file):
    stopwordList = set()
    stopwordFile = open(stopword_file, "r")
    for line in stopwordFile:
        stopwordList.add(line[:-1])
    return stopwordList


def parse_file(file, bloblist, stopwordList):
    not_final = {}
    for line in file:
        tmp = line.split(' ')
        name = ""
        for i in range(len(tmp)):
            if tmp[i] == '"file_name":':
                name = tmp[i + 1][1:-2]
        rstart = line.find("caption")
        r = line[rstart + 10:].lower()
        for char in r:
            if not char.isalnum() and char != ' ':
                r = r.replace(char, '')
        r = r.split(' ')
        newR = ""
        wordSet = set()
        for word in r:
            if word not in stopwordList:
                newR = newR + word + " "
                wordSet.add(word)
        bloblist.append(tb(newR))
        not_final[name] = wordSet
    return not_final


def compute_blobs(bloblist, scores):
    for i, blob in enumerate(bloblist):
        print("Working on document {}".format(i + 1))
        for word in blob.words:
            scores[word] = compute_tfidf(word, blob, bloblist)
    return print("done.")


def create_dic(not_final, scores, final):
    for name in not_final.keys():
        tmpDic = {}
        for word in not_final[name]:
            if word in scores.keys():
                tmpDic[word] = scores[word]
        final[name] = tmpDic
    return


def write_to_file(final, output_file, treshold):
    output = open(output_file, "w")
    for name in final.keys():
        output.write(name + "\t")
        threshold = np.percentile(list(final[name].values()), 100 - int(treshold))
        for word in final[name].keys():
            if final[name][word] >= threshold:
                output.write(word + ":" + str(final[name][word]) + ", ")
        output.write("\n")
    return final


def tfidf(input_file, stopword_file, output_file, treshold):
    file = open(input_file, "r")
    stopwordList = load_stopfile(stopword_file)
    bloblist = []
    scores = {}
    not_final = parse_file(file, bloblist, stopwordList)
    compute_blobs(bloblist, scores)
    final = {}
    create_dic(not_final, scores, final)
    return write_to_file(final, output_file, treshold)


result = tfidf(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
print(result)
