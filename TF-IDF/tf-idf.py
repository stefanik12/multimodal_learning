import math
import sys
from textblob import TextBlob as tb
import statistics

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)


def idf(word, bloblist):
    tmp1 = len(bloblist)
    tmp2 = (1 + n_containing(word, bloblist))
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))


def tfidf(word, blob, bloblist):
    tmp1 = tf(word, blob)
    tmp2 = idf(word, bloblist)
    return tf(word, blob) * idf(word, bloblist)


nFinal = {}
file = open(sys.argv[1], "r")
stopwordList = set()
stopwordFile = open(sys.argv[2], "r")
for line in stopwordFile:
    stopwordList.add(line[:-1])
bloblist = []
for line in file:
    tmp = line.split(' ')
    name = ""
    category = ""
    for i in range(len(tmp)):
        if tmp[i] == '"file_name":':
            name = tmp[i + 1][1:-2]
        if tmp[i] == '"category":':
            category = tmp[i + 1][1:-2]
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
    nFinal[name] = [category, wordSet]

scores = {}
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    #scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    for word in blob.words:
        scores[word] = tfidf(word, blob, bloblist)
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))


final = {}
for name in nFinal.keys():
    tmpDic = {}
    for word in nFinal[name][1]:
        if word in scores.keys():
            tmpDic[word] = scores[word]
    final[name] = (nFinal[name][0], tmpDic)

output = open("output.txt", "w")
for name in final.keys():
    output.write(name +"\t"+ final[name][0]+"\t")
    threshold = statistics.median(final[name][1].values())
    for word in final[name][1].keys():
        if final[name][1][word] >= threshold:
            output.write(word +":" + str(final[name][1][word])+ ", ")
    output.write("\n")

