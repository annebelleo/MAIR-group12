import Levenshtein as lev
import pandas as pd
import itertools
import data_preparation as dp
# TODO: how to handle "any"
#       types doesnt exists in DB
#       center, centre
#       thai vs Thailand
#       oriental
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

def test():
    inform_sentences = dp.get_data()
    inform_sentences = inform_sentences.loc[inform_sentences['label'] == 6]
    data = load_restaurant_data()


    weird_sentence = []
    for s in inform_sentences["sentence"]:
        is_weird = True
        for word in s.split():
            for key in data.keys():
                if word in data[key].to_list():
                    is_weird = False
        if is_weird:
            weird_sentence.append(s)
    txt = ""
    for ws in weird_sentence:
        txt = txt + ws +" " + str(get_preference(ws)) + "\n"

    with open('res/test.txt', 'w') as file:
        file.write(txt)


