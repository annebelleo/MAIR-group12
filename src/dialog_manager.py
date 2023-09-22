# Library and project imports
from data_preparation import class2descript
import pandas as pd
from preference_extraction import get_preference
#import preference_extraction
import models.feed_forward as ffn
import numpy as np

# Setting up data structures
df_restaurant = pd.read_csv('res/restaurant_info.csv')

frame_suggestion = pd.DataFrame(columns=df_restaurant.columns)
list_suggestion = []
frame_user_input = {"area": None,
                    "food": None,
                    "pricerange": None}
list_turns = []

class Suggestion_Manager:
    #config
    dontcarevalue = 'any'
    #control
    suggestions_initialized = False #Use to control when to populate suggestion list
    #data
    suggestion_dicts = [] # Stores a set of suggestions in dictionary form
    suggestion_fields = [] # Stores column names of suggestion for input verification
    suggestion_current = dict() # Stores the current suggestion retreived from the list above.
        
    def load_suggestions(self, input_frame : dict, path = 'res/restaurant_info.csv'):
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
        if input_frame['food'] != dontcarevalue:
            query += f'& food == "{input_frame["food"]}"'
        if input_frame["area"] != dontcarevalue:
            query += f'& area == "{input_frame["area"]}"'
        if input_frame["pricerange"] != dontcarevalue:
            query += f'& pricerange == "{input_frame["pricerange"]}"'
        query = query[1:]
        if query == "":
            df_suggestions = pd.concat([df_suggestions,  df_restaurant.copy()])

        else:
            df_suggestions = pd.concat([df_suggestions, df_restaurant.query(query)])  
        
        # Turns dataframe query into list of dictionaries
        self.suggestion_dicts = df_suggestions.to_dict('records')
        self.suggestions_initialized = True
        return
     
    def propose_suggestion(self):
        if len(self.suggestion_dicts) > 0:
            self.suggestion_current = self.suggestion_dicts.pop(0)
        else:
            self.suggestion_current = None
        return self.suggestion_current
        
    def get_suggestion_information(self, query : list):
        # Assertion to verify if input is valid
        assert set(query).issubset(set(self.suggestion_fields)), 'Query does not correspond to fields'
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
        return len(self.suggestion_dicts) == 0 and not self.suggestion_current
    
    def initialize(self):
        self.suggestion_dicts = []
        self.suggestion_fields = []
        self.suggestion_current = dict()
        
    def reset_suggestions(self):
        '''
        
        '''
        # USE THIS FUNCTION IF STATE 5 RESULTS IN A COMPLETE DO-OVER
        self.suggestions_initialized = False
        self.initialize()
        
    
    def __init__(self):
        self.initialize()
    
    


# Prediction Model
model = ffn.get_model()

# Config        
messages = {
    's0_welcome' : 'Welcome to the chatbot',
    's1_ask_price' : 'What is your budget?',
    's2_ask_area' : 'Which area do you prefer?',
    's3_ask_food': 'What type of food do you like?',
    's4_suggest_restaurant' : "I have found %s. It is an %s restaurant in the %s part of town that serves %s food.\
                    \nAre you interested in it?",
    's5_give_info' : 'give',
    's6_bye' : 'bye',
    's7_restart' : 'No suggestion found',
    
}
state_list = list(messages.keys())
dialog_act = [  'ack', 
                'affirm', 
                'bye', 	
                'confirm',
                'deny',
                'hello',
                'inform',   
                'negate',
                'null',
                'repeat',
                'reqalts',
                'reqmore',
                'request',
                'restart',
                'thankyou'
]
dontcarevalue = 'any' #redundant
suggestions_initialized = False #Use to control when to populate suggestion list



def state_transition(current_state, user_message = None):
    '''Manages state transitions and actions for a turn in the dialog'''
    
    # Set data
    suggestions = Suggestion_Manager()

    def add_to_user_frame(preferences : dict):
        '''
        updates preferences in frame
        '''
        for key in preferences.keys():
            frame_user_input[key] = preferences[key] # modify frame

    def clear_frame():
        '''
        empties current frame user input frame.
        '''
        global frame_user_input
        frame_user_input =   {"area": None,
                              "food": None,
                              "pricerange": None}
    
    def predict_act(message : str):
        
        prediction = model.predict(pd.Series(message))
        return prediction
        
    # Get Data     
    def get_suggestion_data():
        # Returns tuple with info
        
        return 0
    
    def dialog_response(state : str, data : tuple = ()):
        '''
        Prints message associated with state.
        The data passed 
        '''
        message = messages[current_state]
        if data == None or data == (): # Prevents errors, but should not occur.
            message.replace('%s', 'NULL')
        print(message % data)


    # Dialog flow control
    def is_current_state(condition_state : str):
        '''
        Logical boolean for state conditional statements
        '''
        assert current_state in state_list
        return current_state == condition_state
    
    def is_current_label(dialog_act_no : int):
        '''
        Logical boolean for label conditional statements
        '''
        return prediction == dialog_act_no
    
    def is_area_expressed():
        return not frame_user_input['area'] == None

    def is_food_expressed():
        return not frame_user_input['food'] == None

    def is_pricerange_expressed():
        return not frame_user_input['pricerange'] == None



    def process_current_state(state = current_state, label : int = -1):
        '''
        Executes a set of actions that occur for every state.    
        '''
        ### DEV NOTE: Processing is entirely dependent on the action determined by user input,
        ###           and posibilities are limited based on the current state of the dialog.
        ###           needs some work to figure out the logical flow of this.
        dialog_response()
        user_message = input()
        preference = get_preference(user_message)
        add_to_user_frame(preference)

    

    def get_next_state():
        print(current_state)
        if is_current_state('s0_welcome'):
            process_current_state()
            return 's1_ask_price'
            
        elif is_current_state('s1_ask_price'):
            if not is_pricerange_expressed():
                process_current_state()
                return 's1_ask_price'
            return 's2_ask_area'
            
        elif is_current_state('s2_ask_area'):
            if not is_area_expressed():
                process_current_state()
                return 's2_ask_area'
            return 's3_ask_food'
        
        elif is_current_state('s3_ask_food'):
            if not is_food_expressed():
                process_current_state()
                return 's3_ask_food'
            return 's4_suggest_restaurant'
    
        elif is_current_state('s4_suggest_restaurant'):
           
            suggestions.load_suggestions(path = 'res/restaurant_info.csv')
            suggestions.propose_suggestion()
            if suggestions.is_suggestions_exhausted():
                suggestions.reset_suggestions()
                return 's7_restart'
            else:
                # NOTE: Example of querying the data from the current suggestion
                suggestion_data = suggestions.get_suggestion_information(["restaurantname","pricerange","area","food"])
                # NOTE: printing with explicit messages
                print("I have found %s. It is an %s restaurant in the %s part of town that serves %s food.\
                    \nAre you interested in it?" % suggestion_data)
                # NOTE: printing using built in dialog response function.
                # NOTE: Refer to messages dictionary to check and set messages for each state.
                dialog_response(state = current_state, data = suggestion_data)
                
            
            user_message = input()
            label = predict_act(user_message)
            print(label)
            # get clasification
            if label == 1:
                return 's5_give_info'
            elif label == 10:
                return 's4_suggest_restaurant'
            else:
                return 's6_bye'
    
        elif is_current_state('s5_give_info'):
            print("Which information do you want: Phone number, ") #PRINTMESSAGE

        return 's6_bye'
    
    if is_current_state('s6_bye'):
        return 
    frame_user_input["area"]= "centre"
    frame_user_input["food"]= "italian"
    frame_user_input["pricerange"]= "cheap"
    
    # load_suggestions()
    # #print_message()    
   # user_message = input()
            
    next_state = get_next_state()
    state_transition(next_state)

if __name__ == '__main__':
    
    state = state_transition('s0_welcome')
    print(state)

