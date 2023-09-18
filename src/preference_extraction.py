import Levenshtein as lev
import pandas as pd
import itertools
def load_restaurant_data(file_path = 'res/restaurant_info.csv'):
    df = pd.read_csv(file_path)
    # there are some NaN comming in area
    # issue multiple word
    result = {}
    result["pricerange"] = df["pricerange"].drop_duplicates()
    result["area"] = df["area"].drop_duplicates()
    result["food"] = df["food"].drop_duplicates()
    
    result["pricerange"] = result["pricerange"].dropna(axis = 0)
    result["area"] = result["area"].dropna(axis = 0)
    result["food"] = result["food"].dropna(axis = 0)
    
    return result
def get_combinations(sentence):
    sentence = sentence.split()
    xs = [sentence[i:j] for i, j in itertools.combinations(range(len(sentence)+1), 2)]
    output = []
    for i in xs:
        output.append(" ".join(i))
    return output
def get_preference(sentence):
    restaurant_data = load_restaurant_data()
    xs = get_combinations(sentence)
    result = {}
    for word in xs:
        for key in restaurant_data.keys():
            for item in restaurant_data[key]:
                max_distance = len(word)//3
                if lev.distance(word, item) < max_distance:
                    if key not in result:
                        result[key] = []
                        result[key].append(item)
        
    return result

print(get_preference("i want acheap azian orientil food in the center of town"))
