from data_preparation import class2descript
import pandas as pd
from preference_extraction import get_preference
#import preference_extraction
import models.feed_forward as feed_forward
import numpy as np
db = pd.read_csv('res/restaurant_info.csv')
model = feed_forward.get_trained_model()
frame_suggestion = pd.DataFrame(columns=db.columns) # its empty
dontcarevalue = 'any'

frame_user_input = {"area": None,
                    "food": None,
                    "pricerange": None}
                
messages = {
    's0_welcome' : 'Welcome to the chatbot',
    's1_ask_price' : 'What is your budget?',
    's2_ask_area' : 'Which area do you prefer?',
    's3_ask_food': 'What type of food do you like?',
    's4_suggest_restaurant' : 'I suggest',
    's5_give_info' : 'give',
    's6_bye' : 'bye',
    's7_restart' : 'No suggestion found',
    
}

state_list = list(messages.keys())


def state_transition(state, user_message = None):
    def load_suggestions():
        global frame_suggestion 
        for _,value in frame_user_input.items():
            if not value:
                raise Exception('User input not complete') 
        query = ""
        if frame_user_input['food'] != dontcarevalue:
            query += f'& food == "{frame_user_input["food"]}"'
        if frame_user_input["area"] != dontcarevalue:
            query += f'& area == "{frame_user_input["area"]}"'
        if frame_user_input["pricerange"] != dontcarevalue:
            query += f'& pricerange == "{frame_user_input["pricerange"]}"'
        query = query[1:]
        if query == "":
            frame_suggestion = pd.concat([frame_suggestion,  db.copy()])

        else:
            frame_suggestion = pd.concat([frame_suggestion, db.query(query)])  
        

    def is_current_state(condition_state : str):
        assert state in state_list
        return state == condition_state
    
    def current_prediction(dialog_act_no : int):
        return prediction == dialog_act_no
    
    def predict_act(sentence : str):
        # Not implemented yet. Use external library
        return 6
        
    def add_to_user_frame(preferences : dict):
        '''
        updates preferences in frame
        '''
        for key in preferences.keys():
            frame_user_input[key] = preferences[key] # modify frame
            
    def get_frame_from():
        # Returns tuple with info
        return 0
    
    def get_statemsg(state : str):
        messages[state]
        return
    
    def clear_frame():
        '''
        empties current frame
        '''
        frame =  dict(zip(frame_suggestion, [None]*len(frame_suggestion)))
        
    def is_area_expressed():
        return not frame_user_input['area'] == None

    def is_food_expressed():
        return not frame_user_input['food'] == None

    def is_pricerange_expressed():
        return not frame_user_input['pricerange'] == None

    def print_message():
        print(messages[state])


    def process_current_state():
        print_message()
        user_message = input()
        preference = get_preference(user_message)
        add_to_user_frame(preference)

    

    def get_next_state():
        print(state)
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
            global frame_suggestion
            if frame_suggestion.empty: 
                load_suggestions()
            suggestion = frame_suggestion.iloc[0]
            if suggestion.empty:
                return 's7_restart'
            frame_suggestion = frame_suggestion.drop([suggestion.name])   
            print(f"I have found {suggestion.restaurantname}. Are you interested in it?")
            user_message = input()
            label = model.predict(pd.Series(user_message))
            print(label)
            # get clasification
            if label == 1:
                return 's5_give_info'
            elif label == 10:
                return 's4_suggest_restaurant'
            else:
                return 's6_bye'
    
        elif is_current_state('s5_give_info'):
            print("Which information do you want: Phone number, ")

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

