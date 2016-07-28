import json

def main():
	x = 0
	N = 0
	with open('out.json') as data_file:
		data = json.load(data_file)
	for entry in data:
		if entry["r"] == 0.5:
			N += 1
			x += entry["F-Bcubed"]
	print(x/N)

if __name__ == '__main__':
	main()