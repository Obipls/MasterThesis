# !/usr/bin/env python3.5
# coding=utf-8
# http://www.clips.ua.ac.be/sites/default/files/overviewpaper.pdf
# https://cs224d.stanford.edu/reports/YaoLeon.pdf
# http://arxiv.org/ftp/arxiv/papers/1506/1506.04891.pdf


import os, re, sys, ujson, unicodedata
from collections import defaultdict, Counter
from keras.preprocessing.text import Tokenizer


def preprop(token,greek):
    # 141.728 preprocessing replacements (words)
    # Replace all quotes to a single one (11500)
    token = token.replace("‘", '"').replace("’", '"').replace('“', '"').replace('”', '"').replace("'", '"').replace("'",
                                                                                                                  '"').replace(
        '"', '"').replace('¨', '"').replace("´", "'").replace("`", "'")
    # Replace apostrophes with standard ones (0)
    token = token.replace("’", "'").replace("'", "'")
    # Replace commas(0)
    token = token.replace('、', ',').replace('،', ',')
    # Replace dashes(2000)
    token = token.replace('‒', '-').replace('–', '-').replace('—', '-').replace('―', '-')
    # Replace ellipsis(0)
    token = token.replace('. . .', '…').replace('...', '…').replace('⋯', '…')
    # Normalize according to paper(120.000)
    token = unicodedata.normalize('NFKD', token)
    # Remove additional whitespace
    token = re.sub('\s+', ' ', token).strip()
    # Replace numbers with 7
    token = re.sub('\d+', '7', token)
    # If language is greek, change latin token for placeholder (s for convenience)
    if greek:
        m=re.match('[a-zA-Z]',token)
        if m:
            print(token)
        token = re.sub('[a-zA-Z]','s',token)
        if m:
            print(token)

    return token


def main():
    occurenceDict = defaultdict(list)
    with open('data/info.json') as j:
        info = ujson.load(j)
    for problem in os.listdir('data'):
        greek=False
        if problem.startswith('problem'):
            print(problem)
            probTokList=[]
            path = 'data/' + problem
            for entry in info:
                if entry["folder"]==problem:
                    if entry["language"]=="gr":
                        greek=True
            for doc in os.listdir(path):
                docTokList = []
                with open(path + '/' + doc) as d:
                    text = d.read()
                    probTokList.append(text)
                    for token in text:
                        procToken=preprop(token,greek)
                        docTokList.append(procToken)
                probTokList.append(' '.join(docTokList))
            #print(probTokList[0])
            #Tokenizer.texts_to_sequences(probTokList)




                        #for i, token in enumerate(' '.join(linesplit)):
                         #   occurenceDict[token].append(i)
                          #  token = occurenceDict[token][0]
                           # docTokList.append(token)
                #print(docTokList)


if __name__ == '__main__':
    main()
