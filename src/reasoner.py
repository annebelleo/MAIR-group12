import pytholog as pl
import pandas as pd
from re import sub, findall
def camel_case(s):
  s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
  return ''.join([s[0].lower(), s[1:]])

class Reasoner:
    def __init__(self):
        self.init_knowledge_base()
    def init_knowledge_base(self):
        self.knowledge_base = pl.KnowledgeBase("restaurants")
        self.knowledge_base(["touristic(X) :- restaurant(X,good,_,_,_,cheap)",
                "notTouristic(X) :- restaurant(X,_,_,_,romanian,_)",
                "assignedSeats(X) :- restaurant(X,_,busy,_,_,_)",
                "notRomantic(X) :- restaurant(X,_,busy,_,_,_)",
                "notChildren(X) :- restaurant(X,_,_,long,_,_)",
                "romantic(X) :- restaurant(X,_,_,long,_,_)",
                "neq(touristic(X),notTouristic(X))",
                "neq(notRomantic(X),romantic(X))"])
    def query(self,query,find_all = False):
        if find_all:
            return self.knowledge_base.query(pl.Expr(query),show_path=True)
        else:
            res = findall(r'\(.*?\)', query)
            res = res[0][1:-1]
            query =query.replace(res,camel_case(res))
            return self.knowledge_base.query(pl.Expr(query),show_path=True)
        
    def add_facts(self,df):
        for i in range(df.shape[0]):
            # restaurant(Name,quality,crowd,length,food,price)
            x = f"restaurant({camel_case(df.restaurantname[i])},{camel_case(df.food_quality[i])},{camel_case(df.crowdedness[i])},{camel_case(df.length_of_stay[i])},{camel_case(df.food[i])},{camel_case(df.pricerange[i])})"
            self.knowledge_base([x])


def filter(prefiltered_list,filters):
    output= []
    
    reasoner = Reasoner()
    reasoner.add_facts(pd.read_csv('res/restaurant_extra_info.csv'))
    for x in range(prefiltered_list.shape[0]):
        q=reasoner.query(f"{filters}({prefiltered_list.iloc[x].restaurantname}))")
        if q[0]==["Yes"]:
           # output = pd.concat([output, prefiltered_list.iloc[x]], join='inner')
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
def get_reasoning(q):
    rule_format = ["name","quality","crowd","lengthOfStay","foodType","price"]
    reasoner= Reasoner()
    rule = str(reasoner.knowledge_base.db[q]["facts"][0])
    rule = rule.split("(")[-1][:-1].split(",")
    output = dict()
    for i in range(len(rule)):
        if rule[i] !="_":
            output.update({rule_format[i]:rule[i]})
    output.update({"rule":q})
    string = f'It is {un_camel_caseify(output["rule"])} because '
    keys = list(output.keys())
    if len(output)==3:
        string = string + f'the {un_camel_caseify(keys[1])} is {output[get_nth_key(output,1)]}'
    elif len(output)==4:
        string = string + f'the {un_camel_caseify(keys[1])} is {output[get_nth_key(output,1)]} and the {un_camel_caseify(keys[2])} is {output[get_nth_key(output,2)]}'
    else:
        for x in range(1,len(output)-1):
            string = string + f'the {un_camel_caseify(keys[x])} is {output[get_nth_key(output,x)]}, '
    return string


if __name__ == '__main__':
    
    l = pd.read_csv('res/restaurant_extra_info.csv')
    q = "notTouristic"
    print(filter(l, q))
    print(get_reasoning(q))
