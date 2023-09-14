import sklearn
import pandas as pd
from data_preparation import get_data
import matplotlib.pyplot as plt
import numpy as np
def rule_baseline(data_df):
    token_df = data_df["tokenized"]
    output = []
    for value in token_df:
        if "okay" in value:
            output.append(0)
        elif "yes" in value:
            output.append(1)
        elif "bye" in value:
            output.append(2)
        elif "is" in value:
            output.append(3)
        elif "dont"in value or "not" in value:
            output.append(4)
        elif "hi" in value or "hello" in value:
            output.append(5)
        elif "need" in value or "want" in value:
            output.append(6)
        elif "no" in value:
            output.append(7)
        elif "uhhh" in value or "noise" in value or "cough" in value:
            output.append(8)
        elif "repeat" in value or "again" in value:
            output.append(9)
        elif "how" and "about" in value:
            output.append(10)
        elif "more" in value:
            output.append(11)
        elif "what" in value:
            output.append(12)
        elif "restart" in value or ("start" and "over") in value :
            output.append(13)
        elif "thank" in value or "thanks" in value:
            output.append(14)
        else:
            output.append(-1)
    return output
train, test = sklearn.model_selection.train_test_split(get_data(),test_size=0.15)
predicted = rule_baseline(test)
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(predicted)
axs[1].hist(test["class"])
plt.show()
print("Accuracy: " + str(np.mean(predicted==test["class"])))