import requests
import json

completeLabelsPath = "classification/labels.json"
samplePath = "../labels/names_test.json"
with open(samplePath, 'r') as file:
    data = json.load(file)
    
species = data.keys()
names = {}
named_count = 0
non_named_count = 0
var = r'\\'
for fish in species:
    try:
        number = 0
        if var[0] in data[fish][number]["vernacularName"]:
            number +=1
        names[fish] = data[fish][number]["vernacularName"]
        named_count += 1
    except:
        names[fish] = fish
        non_named_count += 1 
        continue
print(names)
print("Named Count:", named_count)
print("Non Named count: ", non_named_count)
print(var[0])


json_object = json.dumps(names, indent=4)

with open("../labels/vernacular_names.json", "w") as outfile:
    outfile.write(json_object)
    
#FETCH FROM GBIF

mainUrl = "https://api.gbif.org/v1/species/search?datasetKey=d7dddbf4-2cf0-4f39-9b2a-bb099caae36c&q="


vernacularNames = {}
for fish in species:
    print(fish)
    try:
        res = requests.get(mainUrl+fish)
        response = json.loads(res.text)
        vernacularNames[fish] = response["results"][0]["vernacularNames"]
        
    except:  
        continue

json_object = json.dumps(vernacularNames, indent=4)

with open("../labels/names_test.json", "w") as outfile:
    outfile.write(json_object)
    