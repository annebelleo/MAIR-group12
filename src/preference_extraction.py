import Levenshtein as lev
import pandas as pd
def load_restaurant_data(file_path = 'res/restaurant_info.csv'):
    df = pd.read_csv(file_path)
    # there are some NaN comming in area
    
    result = {}
    result["pricerange"] = df["pricerange"].drop_duplicates()
    result["area"] = df["area"].drop_duplicates()
    result["food"] = df["food"].drop_duplicates()
    
    result["pricerange"] = result["pricerange"].dropna(axis = 0)
    result["area"] = result["area"].dropna(axis = 0)
    result["food"] = result["food"].dropna(axis = 0)
    
    return result

def get_preference(sentence):
    restaurant_data = load_restaurant_data()
    result = {}
    for word in sentence.split():
        for key in restaurant_data.keys():
            for item in restaurant_data[key]:
                if lev.distance(word, item) < 3:
                    print(word, item)
                    if key not in result:
                        result[key] = []
                        result[key].append(item)
        
    return result

print(get_preference("i want thai food"))