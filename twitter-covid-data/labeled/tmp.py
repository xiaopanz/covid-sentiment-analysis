import json

with open ('2020-07-labeled.json', 'r') as fin:
	read_in = fin.readlines()

data = []
for el in read_in:
	data.append(json.load(el))

ret = []
with open ('2020-07-labeled-1.json', 'w') as fout:
	for el in data:
		d = {}
		d['text'] = el['test']
		d['label'] = el['label']
		ret.append(d)
	fout.write(json.dump(ret, indent=4, sort_keys=True))
