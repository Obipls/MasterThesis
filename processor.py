# !/usr/bin/env python3.5
# coding=utf-8
# http://www.clips.ua.ac.be/sites/default/files/overviewpaper.pdf
# https://cs224d.stanford.edu/reports/YaoLeon.pdf
# http://arxiv.org/ftp/arxiv/papers/1506/1506.04891.pdf

import os, re, sys, ujson, unicodedata
from keras.preprocessing import text, sequence
from NNcompare import sharedNN, embedNN
from itertools import combinations,permutations
from collections import defaultdict, Counter
from progressbar import ProgressBar,Timer,Bar,ETA
from clusterer import KNNclusterer, MSclusterer
import numpy as np

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
			docMatrix = tokenizer.texts_to_matrix(probTokList)
			seqList = tokenizer.texts_to_sequences(probTokList)
			uniqueTokens = max([max(x) for x in seqList])
			print(uniqueTokens,lang)
			sampling_table = sequence.make_sampling_table(uniqueTokens+1)
			for i,seq in enumerate(seqList):
				x, y = sequence.skipgrams(seq, uniqueTokens, window_size=4, negative_samples=1., categorical=True, sampling_table=sampling_table)
				X.extend(x)
				Y.extend(y)
				docDict[docList[i]] = seq
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
			truthCounter =  Counter(nnDict.values())
			baseline = 1-float(truthCounter[True])/float(len(nnDict))
			print("Baseline for {} is {}".format(problem, baseline))
			clusterCount = Counter()
			for nclusters in reversed(range(len(docMatrix)-1)):
				#print("{} Clusters".format(nclusters+1))
				#clusters = KNNclusterer(nclusters+1,docMatrix)
				clusters = MSclusterer(docMatrix)
				for c in range(nclusters+1):
					#print(c,"has:",[i for i,x in enumerate(clusters) if x == c])
					for clusterpair in list(combinations([i for i,x in enumerate(clusters) if x == c],2)):
						combo = (docList[clusterpair[0]],docList[clusterpair[1]])
						clusterCount[combo] +=1
			#print(cList)
			x = 0.0 
			scoreList[0] += truthCounter[True]
			for combo in clusterCount.most_common(20):
				if combo[0] in cList:
					x += 1
					scoreList[1] += 1
			print("Document score is {} clusters correct out of {} ({})".format(x, truthCounter[True], x/truthCounter[True]))
	print("Total score  is {}, {} clusters correct".format(scoreList[1]/scoreList[0], scoreList[1]))


			#scores = sharedNN(docDict,nnDict)

						# pairscore = LSTMcomp((docDict[pair[0]],docDict[pair[1]]),match)
						# scoreDict[pair[0]].append((pair[1],pairscore))
						# scoreDict[pair[1]].append((pair[0],pairscore))


				# clusters = cluster_docs(scoreDict,docDict)
				# jsonList=[]
				# for cluster in clusters:
				#     cList=[]
				#     for doc in cluster:
				#         cList.append({"document":doc})
				#     jsonList.append(cluster)
				# with open("clustering.json", "w") as jsonFile:
				#     json.dumps(jsonList.jsonFile)






if __name__ == '__main__':
	main()
