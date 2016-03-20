#!/usr/bin/env python
#http://www.clips.ua.ac.be/sites/default/files/overviewpaper.pdf 
#https://cs224d.stanford.edu/reports/YaoLeon.pdf
#http://arxiv.org/ftp/arxiv/papers/1506/1506.04891.pdf


import os, sys

def main():
	for problem in os.listdir('data'):
		if problem.startswith('problem'):
			print(problem)
			path='data/'+problem
			for doc in os.listdir(path):
				with open(path+'/'+doc) as d:
					text=d.read()#.splitlines()
					


if __name__ == '__main__':
	main()