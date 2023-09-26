import pytholog as pl
import pandas as pd
from re import sub, findall
def camel_case(s):
  s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
  return ''.join([s[0].lower(), s[1:]])

class Reasoner:
    def __init__(self):
        self.reset_kb()
    def reset_kb(self):
        self.new_kb = pl.KnowledgeBase("restaurants")
        self.new_kb(["touristic(X) :- restaurant(X,good,_,_,_,cheap)",
                "notTouristic(X) :- restaurant(X,_,_,_,romanian,_)",
                "assignedSeats(X) :- restaurant(X,_,busy,_,_,_)",
                "notRomantic(X) :- restaurant(X,_,busy,_,_,_)",
                "notChildren(X) :- restaurant(X,_,_,long,_,_)",
                "romantic(X) :- restaurant(X,_,_,long,_,_)",
                "neq(touristic(X),notTouristic(X))",
                "neq(notRomantic(X),romantic(X))"])
    def query(self,query,find_all = False):
        if find_all==True:
            return self.new_kb.query(pl.Expr(query))
        else:
            res = findall(r'\(.*?\)', query)
            res = res[0][1:-1]
            query =query.replace(res,camel_case(res))
            return self.new_kb.query(pl.Expr(query))
    def addFacts(self,df):
        for i in range(df.shape[0]):
            # restaurant(Name,quality,crowd,length,food,price)
            x = f"restaurant({camel_case(df.restaurantname[i])},{camel_case(df.food_quality[i])},{camel_case(df.crowdedness[i])},{camel_case(df.length_of_stay[i])},{camel_case(df.food[i])},{camel_case(df.pricerange[i])})"
            self.new_kb([x])
if __name__ == '__main__':
    test = Reasoner()
    test.addFacts(pd.read_csv('res/restaurant_extra_info.csv'))
    print(test.query("touristic(the missing sock))")) 