import Levenshtein as lev
import pandas as pd
import itertools
import data_preparation as dp
# TODO: 
#       how to handle "any"
#       types doesnt exists in DB
alternate_keywords = {"thailand":"thai",
                      "oriental":"asian oriental",
                      "asian":"asian oriental",
                      "gastro":"gastropub",
                      "hindi":"indian",
                      "american":"northamerican",
                      "steak":"steakhouse",
                      "turkey":"turkish",
                      "turk":"turkish",
                      "vietnam":"vietnamese",
                      "center":"centre",
                      "central":"centre",
                      "northern":"north",
                      "eastern":"east",
                      "western":"west",
                      "southern":"south"
                      }
any_alternate_keywords = {"care":"any","anything":"any"}
def load_restaurant_data(category, file_path = 'res/restaurant_info.csv'):
    df = pd.read_csv(file_path)
    result = {}
    if category == "pricerange":
        result["pricerange"] = df["pricerange"].drop_duplicates()
        result["pricerange"] = result["pricerange"].dropna(axis = 0)
        result["pricerange"]= pd.concat([result["pricerange"],pd.Series(["any"])],ignore_index=True)
    elif category == "area":
        result["area"] = df["area"].drop_duplicates()
        result["area"] = result["area"].dropna(axis = 0)
        result["area"]= pd.concat([result["area"],pd.Series(["any"])],ignore_index=True)
    elif category == "food":
        result["food"] = df["food"].drop_duplicates()
        result["food"] = result["food"].dropna(axis = 0)
        result["food"]= pd.concat([result["food"],pd.Series(["any"])],ignore_index=True)
    elif category == None:
        result["food"] = df["food"].drop_duplicates()
        result["pricerange"] = df["pricerange"].drop_duplicates()
        result["area"] = df["area"].drop_duplicates()

        result["food"] = result["food"].dropna(axis = 0)        
        result["pricerange"] = result["pricerange"].dropna(axis = 0)
        result["area"] = result["area"].dropna(axis = 0)

    return result

def get_combinations(sentence):
    sentence = sentence.split()
    xs = [sentence[i:j] for i, j in itertools.combinations(range(len(sentence)+1), 2)]
    output = []
    for i in xs:
        output.append(" ".join(i))
    return output

def get_preference(sentence, categories = None):
    dict = {}
    if categories:
      dict = {**alternate_keywords,**any_alternate_keywords}
    else:
      dict = alternate_keywords
    for input_word, alt in dict.items():
        for lev_word in sentence.split():
            if lev.distance(input_word,lev_word)<2:
                sentence= sentence.replace(lev_word,alt)

    restaurant_data = load_restaurant_data(category= categories)
    xs = get_combinations(sentence)
    result = {}
    for word in xs:
        for key in restaurant_data.keys():
            for item in restaurant_data[key]:
                max_distance = len(word)//3
                if lev.distance(word, item) < max_distance:
                    if key not in result:
                        result[key]=(item)
    return result

def test(categories = None):
    inform_sentences = dp.get_data()
    inform_sentences = inform_sentences.loc[inform_sentences['label'] == 6]
    data = load_restaurant_data(category=categories)


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
        txt = txt + ws +" " + str(get_preference(ws,categories=categories)) + "\n"

    with open('res/test.txt', 'w') as file:
        file.write(txt)
