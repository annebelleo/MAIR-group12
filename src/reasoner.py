import pytholog as pl
import pandas as pd
from re import sub, findall
import system_messages
def camel_case(s):
  s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
  return ''.join([s[0].lower(), s[1:]])

class Reasoner: # Main reasoner class. Representation of KB is stored here
    def __init__(self):
        self.init_knowledge_base()
    def init_knowledge_base(self): # Initialization of rules specified in task 1c
        self.knowledge_base = pl.KnowledgeBase("restaurants")
        self.knowledge_base(["touristic(X) :- restaurant(X,good,_,_,_,cheap)", #Ex. X is touristic if it has good food, is cheap and all other values are irrelavant
                "notTouristic(X) :- restaurant(X,_,_,_,romanian,_)",
                "assignedSeats(X) :- restaurant(X,_,busy,_,_,_)",
                "notRomantic(X) :- restaurant(X,_,busy,_,_,_)",
                "notChildren(X) :- restaurant(X,_,_,long,_,_)",
                "romantic(X) :- restaurant(X,_,_,long,_,_)",
                "neq(touristic(X),notTouristic(X))", # Negation. X cannot be touristic and not touristic
                "neq(notRomantic(X),romantic(X))"])
    def query(self,query,find_all = False): # Query function
        if find_all: # If wants to query with variables
            return self.knowledge_base.query(pl.Expr(query),show_path=True)
        else: # if wants a true false query. In this case all uppercase first characters are turned to camelcase to represent constants
            res = findall(r'\(.*?\)', query)
            res = res[0][1:-1]
            query =query.replace(res,camel_case(res))
            return self.knowledge_base.query(pl.Expr(query),show_path=True)
        
    def add_facts(self,df): # Function to add new fact to the kb
        for i in range(df.shape[0]):
            # restaurant(Name,quality,crowd,length,food,price)
            x = f"restaurant({camel_case(df.restaurantname[i])},{camel_case(df.food_quality[i])},{camel_case(df.crowdedness[i])},{camel_case(df.length_of_stay[i])},{camel_case(df.food[i])},{camel_case(df.pricerange[i])})"
            self.knowledge_base([x])


def filter(prefiltered_list,filters): # Get all restaurants that satisfy a rule as specified in filter
    output= []
    
    reasoner = Reasoner()
    reasoner.add_facts(pd.read_csv('res/restaurant_extra_info.csv'))
    for x in range(prefiltered_list.shape[0]):
        q=reasoner.query(f"{filters}({prefiltered_list.iloc[x].restaurantname}))")
        if q[0]==["Yes"]:
            output.append(prefiltered_list.iloc[x])
    return output

def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")

def un_camel_caseify(string):
    return sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', string).lower()

def get_reasoning(q,style): # String manipulation to turn the rule applied to a natural language sentence
    rule_format = ["name","quality","crowd","lengthOfStay","foodType","price"]
    reasoner= Reasoner()
    rule = str(reasoner.knowledge_base.db[q]["facts"][0])
    rule = rule.split("(")[-1][:-1].split(",")
    output = dict()
    for i in range(len(rule)):
        if rule[i] !="_":
            output.update({rule_format[i]:rule[i]})
    output.update({"rule":q})
    keys = list(output.keys())
    if len(output)==3:
        return system_messages.MESSAGES["1_factor_reasoning"][style]%(un_camel_caseify(output["rule"]),un_camel_caseify(keys[1]),output[get_nth_key(output,1)])
    elif len(output)==4:
        return system_messages.MESSAGES["2_factor_reasoning"][style]%(un_camel_caseify(output["rule"]),un_camel_caseify(keys[1]),output[get_nth_key(output,1)],un_camel_caseify(keys[2]),output[get_nth_key(output,2)])




if __name__ == '__main__':
    pass