import sklearn
import pandas as pd
from data_preparation import get_data
import matplotlib.pyplot as plt
import numpy as np
def rule_baseline(data_df):
    token_df = data_df["lines"]
    # token_df = ["okay please do"]
    output = []
    dict = {0:["okay","ok","kay","k", "right"],1:["yes"],2:["bye"],3:["is","does"],
            4:["don't","not","wrong"],5:["hi","hello"],6:["cheap", "care", "want","look for","looking"],
            7:["no"],8:["sil","uh","cough","noise","unintelligible"],9:["repeat","again"],
            10:["how","about", "else"],11:["more"],12:["type", "number", "what", "address", "code", "phone", "post"],13:["restart","start over"],14:["good", "thank", "bye", "you"]}
    for token in token_df:
        flag= False
        for cat in dict.values():
            for options in cat:
                if options in token:
                    output.append(list(dict.keys())[list(dict.values()).index(cat)])
                    flag = True
                    break
            if flag==True:
                    break
        if flag == False:
            output.append(6)
    return output

if __name__ == '__main__':
    train, test = sklearn.model_selection.train_test_split(get_data(),test_size=0.15)
    predicted = rule_baseline(test)
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(predicted)
    axs[1].hist(test["class"])
    plt.show()
    print("Accuracy: " + str(np.mean(predicted==test["class"])))