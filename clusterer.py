# !/usr/bin/env python3.5
# coding=utf-8
#http://www.clips.ua.ac.be/sites/default/files/overviewpaper.pdf 
#https://cs224d.stanford.edu/reports/YaoLeon.pdf
#http://arxiv.org/ftp/arxiv/papers/1506/1506.04891.pdf


import os,re,sys,json,unicodedata
from collections import defaultdict,Counter
from keras.preprocessing.text import one_hot

def preprop(word):
	#141.728 preprocessing replacements
	# Replace all quotes to a single one (11500)
	word=word.replace("‘",'"').replace("’",'"').replace('“','"').replace('”','"').replace("'",'"').replace("'",'"').replace('"','"').replace('¨','"').replace("´","'").replace("`","'")
	# Replace apostrophes with standard ones (0)
	word=word.replace("’","'").replace("'","'")
	# Replace commas(0)
	word=word.replace('、',',').replace('،',',')
	# Replace dashes(2000)
	word=word.replace('‒','-').replace('–','-').replace( '—','-').replace('―','-')
	# Replace ellipsis(0)
	word=word.replace('. . .','…').replace('...','…').replace('⋯','…')
	# Normalize according to paper(120.000)
	word=unicodedata.normalize('NFKD', word)
	# Remove additional whitespace
	word=re.sub( '\s+', ' ', word ).strip()
	# Replace numbers with 7
	re.sub('\d+', '7', word)
	return word


def main():
	occurenceDict=defaultdict(list)
	with open('data/info.json') as j:
		info = json.load(j)
	for problem in os.listdir('data'):
		if problem.startswith('problem'):
			print(problem)
			path='data/'+problem
			for doc in os.listdir(path):
				docTokList=[]
				with open(path+'/'+doc) as d:
					text=d.read().splitlines()
					for line in text:
						linesplit=line.strip().split()
						for i,word in enumerate(linesplit):
							linesplit[i]=preprop(word)

						




						for i,token in enumerate(' '.join(linesplit)):
							occurenceDict[token].append(i)
							token=occurenceDict[token][0]
							docTokList.append(token)
				print(docTokList)









if __name__ == '__main__':
	main()