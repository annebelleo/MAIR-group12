import pandas as pd
import rules
class Suggestion_Manager:
    #config
    dontcarevalue = 'any'
    #control
    suggestions_initialized = False #Use to control when to populate suggestion list
    #data
    suggestion_list = [] # Stores a set of suggestions in dictionary form
    suggestion_fields = [] # Stores column names of suggestion for input verification
    suggestion_current = dict() # Stores the current suggestion retreived from the list above.
        
    def load_suggestions(self, input_frame : dict, path = 'res/restaurant_extra_info.csv', rules = ""):
        if self.is_initialized():
            # Skip this function if initialization already done.
            # Use reload to reset state of initialization
            return
        
        df_restaurant = pd.read_csv(path)
        df_suggestions = pd.DataFrame(columns=df_restaurant.columns)
        self.suggestion_fields = [df_suggestions.columns] # Updates field list
        
        # Determines whether user input is complete
        for _,value in input_frame.items():
            if not value:
                raise Exception('User input not complete')
            
        # Performs query on restaurant csv 
        query = ""
        if input_frame['food'] != self.dontcarevalue:
            query += f'& food == "{input_frame["food"]}"'
        if input_frame["area"] != self.dontcarevalue:
            query += f'& area == "{input_frame["area"]}"'
        if input_frame["pricerange"] != self.dontcarevalue:
            query += f'& pricerange == "{input_frame["pricerange"]}"'
        query = query[1:]
        if query == "":
            df_suggestions = pd.concat([df_suggestions,  df_restaurant.copy()])

        else:
            df_suggestions = pd.concat([df_suggestions, df_restaurant.query(query)])  
        
        # Turns dataframe query into list of dictionaries
        self.suggestion_list = df_suggestions.to_dict('records')
        self.suggestions_initialized = True
        return
     
    def propose_suggestion(self):
        if len(self.suggestion_list) > 0:
            self.suggestion_current = self.suggestion_list.pop(0)
        else:
            self.suggestion_current = None
        return self.suggestion_current
        
    def get_suggestion_information(self, query : list):
        # Assertion to verify if input is valid
       # assert set(query).issubset(set(self.suggestion_fields)), 'Query does not correspond to fields'
        assert self.is_initialized, 'please make a suggestion first before asking for information.'
        data = []
        for q in query:
            data.append(self.suggestion_current[q])
            
        return tuple(data)
    
    def is_initialized(self):
        return self.suggestions_initialized
    
    def is_suggestions_exhausted(self):
        '''
        Determines whether the current selection and backlog are exhausted
        '''
        return len(self.suggestion_list) == 0 and not self.suggestion_current
    
    def initialize(self):
        self.suggestion_list = []
        self.suggestion_fields = []
        self.suggestion_current = dict()
        
    def reset_suggestions(self):
        '''
        
        '''
        # USE THIS FUNCTION IF STATE 5 RESULTS IN A COMPLETE DO-OVER
        self.suggestions_initialized = False
        self.initialize()

    def get_number_suggestions(self):
        return len(self.suggestion_list)    
    
    def filter(self, filter):
        self.suggestion_list =  rules.filter(pd.DataFrame(self.suggestion_list), filter)

    
    def __init__(self):
        self.initialize()