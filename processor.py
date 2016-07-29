# !/usr/bin/env python3.5
# coding=utf-8
# http://www.clips.ua.ac.be/sites/default/files/overviewpaper.pdf
# https://cs224d.stanford.edu/reports/YaoLeon.pdf
# http://arxiv.org/ftp/arxiv/papers/1506/1506.04891.pdf

import os, re, sys, ujson, unicodedata, json
from keras.preprocessing import text, sequence
from NNcompare import sharedNN, embedNN
from itertools import combinations,permutations
from collections import defaultdict, Counter
from progressbar import ProgressBar
from clusterer import KMclusterer, MSclusterer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import time


def preprop(token,greek):
	# Replace all quotes to a single one
	token = token.replace("‘", '"').replace("’", '"').replace('“', '"').replace('”', '"').replace("'", '"').replace("'",
																																																								'"').replace(
			'"', '"').replace('¨', '"').replace("´", "'").replace("`", "'")
	# Replace apostrophes with standard ones
	token = token.replace("’", "'").replace("'", "'")
	# Replace commas
	token = token.replace('、', ',').replace('،', ',')
	# Replace dashes
	token = token.replace('‒', '-').replace('–', '-').replace('—', '-').replace('―', '-')
	# Replace ellipsis
	token = token.replace('. . .', '…').replace('...', '…').replace('⋯', '…')
	# Normalize according to paper
	#token = unicode(token, "utf-8")
	#token = unicodedata.normalize('NFKD', token)
	# Remove additional whitespace
	token = re.sub('\s+', ' ', token).strip()
	# Replace numbers with 7
	token = re.sub('\d+', '7', token)
	# If language is greek, change latin token for placeholder (s for convenience)
	if greek:
			token = re.sub('[a-zA-Z]','s',token)
	return token

def main():
	scoreList = [0.0,0.0]
	with open('data/info.json') as j:
		info = ujson.load(j)
	for problem in os.listdir('data'):
		greek=False
		if problem.startswith('problem'):
			truthPath = 'data/truth/'+problem+'/clustering.json'
			with open(truthPath) as t:
				truth = ujson.load(t)
			print(problem)
			probTokList = []
			docList = []
			docDict = {}
			X=[]
			Y=[]

			path = 'data/' + problem
			for entry in info:
				if entry["folder"] == problem:
					lang=entry["language"]
					if entry["language"] == "gr":
						greek=True

			CV = CountVectorizer(input='filename', strip_accents='unicode', analyzer='word', ngram_range=(1,4))
			docs = [path+'/'+x for x in os.listdir(path)]
			cMatrix = CV.fit_transform(docs)
			for doc in os.listdir(path):
				docTokList = []
				with open(path + '/' + doc) as d:
						article = d.readlines()
						for sent in article:
							sentTokList = []
							for word in sent.split():
								for token in word:
									procToken = preprop(token,greek)
									sentTokList.append(procToken) #Every item of the list is a normalized character
							docTokList.append(' '.join(sentTokList))#Every item of the list is a sentence
				probTokList.append(' '.join(docTokList))#Every item of the list is a document
				docList.append(doc)
			tokenizer = text.Tokenizer(nb_words=None,filters=text.base_filter(),lower=True,split=" ")
			tokenizer.fit_on_texts(probTokList)
			seqList = tokenizer.texts_to_sequences(probTokList)
			
			uniqueTokens = max([max(x) for x in seqList])

			print(uniqueTokens,lang)
			sampling_table = sequence.make_sampling_table(uniqueTokens+1)
			for i,seq in enumerate(seqList):
				x, y = sequence.skipgrams(seq, uniqueTokens, window_size=4, negative_samples=1.0, categorical=False, sampling_table=sampling_table)
				x = zip(x, y)
				X.append(x)
				#Y.extend(y)
				docDict[docList[i]] = seq
			strX=[str(x) for x in X]
			xTokenizer = text.Tokenizer(nb_words=None,filters=text.base_filter(),lower=True,split=" ")
			xTokenizer.fit_on_texts(strX)
			#docMatrix = tokenizer.sequences_to_matrix(seqList,mode="tfidf")
			docMatrix = xTokenizer.sequences_to_matrix(strX,mode="tfidf")
			#scores = embedNN(X,Y)
			pairs = combinations(docDict.keys(),2)
			cList = []
			nnDict = {}
			for cluster in truth:
				cPairs = []
				if len(cluster) > 1:
					for item in cluster:
						cPairs.append(str(item["document"]))
					cList.extend(list(permutations(cPairs,2)))
			for pair in pairs:
				match = False
				if pair in cList:
					match = True
				nnDict[pair] = match
			for i, doc in enumerate(docMatrix):
				docDict[docList[i]] = doc

			
			truthCounter =  Counter(nnDict.values())
			baseline = 1-float(truthCounter[True])/float(len(nnDict))
			print("Baseline for {} is {}".format(problem, baseline))
			clusterCount = Counter()
			kmclusters = False # Change to False for meanshift
			if kmclusters:
				pbar = ProgressBar()
				for nclusters in pbar(reversed(range(len(docMatrix)-1))):
					#print("{} Clusters".format(nclusters+1))
					clusters = KMclusterer(nclusters+1,cMatrix)
					for c in range(nclusters+1):
						#print(c,"has:",[i for i,x in enumerate(clusters) if x == c])
						for clusterpair in list(combinations([i for i,x in enumerate(clusters) if x == c],2)):
							combo = (docList[clusterpair[0]],docList[clusterpair[1]])
							clusterCount[combo] +=1
			else:
				clusters = KMclusterer(int(len(docMatrix)*0.67),docMatrix)
				#clusters = MSclusterer(cMatrix)#cMatrixdocMatrix
				for clusterpair in list(combinations([i for i,x in enumerate(clusters)],2)):
					combo = (docList[clusterpair[0]],docList[clusterpair[1]])
					clusterCount[combo] +=1

			x = 0.0 
			scoreList[0] += truthCounter[True]
			deleteList = []
			#print("Most common cluster is in {}%".format((float(clusterCount.most_common(20)[19][1])/len(docMatrix))*100))
			for combo in nnDict.keys():
				if combo not in clusterCount.keys():
					deleteList.append(combo)
			y = 0.0
			for item in deleteList:
				if item in cList:
					y+=1
				del nnDict[item]
			#scores = sharedNN(docDict, nnDict)
			print("Deleted pairs are {}% of total correct pairs, {}% of deleted pairs was wrongly deleted".format(round(y/len(cList)*100.0,2), round(y/len(deleteList)*100.0,2)))

			for combo in clusterCount.most_common(20):
				if combo[0] in cList:
					x += 1
					scoreList[1] += 1
			print("prec: {}".format(x/20))
			#print("Document score is {} clusters correct out of {} (accuracy {})".format(x, truthCounter[True], x/truthCounter[True]))
			#print("prec: {} \nrec: {}".format(x/20, x/len(nnDict.values())))

	#print("Total precision  is {}, {} clusters correct".format(scoreList[1]/scoreList[0], scoreList[1]))


			#scores = sharedNN(docDict,nnDict)

						# pairscore = LSTMcomp((docDict[pair[0]],docDict[pair[1]]),match)
						# scoreDict[pair[0]].append((pair[1],pairscore))
						# scoreDict[pair[1]].append((pair[0],pairscore))

			if not os.path.exists('answers/'+problem):
				os.mkdir('answers/'+problem)
			clusDict = defaultdict(list)
			rankDict = defaultdict(list)
			for i, cluster in enumerate(list(clusters)):
				clusDict[cluster] .append({"document": docList[i]})
				rankDict[cluster] .append(docList[i])
			with open('answers/'+problem+'/clustering.json', "w") as jsonFile:
				ujson.dump(list(clusDict.values()), jsonFile, indent=4)
			rankList = []
			for value in rankDict.values():
				if len(value) > 1 :
					pairs = combinations(value,2)
				for pair in pairs:
					rankList.append({"document1": pair[0], "document2": pair[1], "score":  1})
			with open('answers/'+problem+'/ranking.json', "w") as jsonFile:
				ujson.dump(rankList, jsonFile, indent=4)






if __name__ == '__main__':
	start_time = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - start_time))
