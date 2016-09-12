import json

def main():
	x = 0
	N = 0
	with open('out_baseline.json') as data_file:
		data = json.load(data_file)
	for entry in data:
		N += 1
		x += entry["P-Bcubed"]
	print(x/N)

if __name__ == '__main__':
	main()