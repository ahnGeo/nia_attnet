import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('results_json_path', type=str)

def main(args) :
    with open(args.results_json_path, 'r') as f :
        data = json.load(f)

    results = []

    for line in data['scenes'] :
        if not line['objects'] == [] :
            if line['objects'][0]["situation"] == line['objects'][0]['gt_situation'] :
                results.append(1)
            else :
                results.append(0)
   
    print(results.count(1) / len(results))
    
if __name__ == '__main__' : 
    args = parser.parse_args()
    main(args)
