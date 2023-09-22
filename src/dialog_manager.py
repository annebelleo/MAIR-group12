from data_preparation import class2descript
import pandas as pd
from preference_extraction import get_preference
#import preference_extraction
import models.feed_forward as ffn
import numpy as np
db = pd.read_csv('res/restaurant_info.csv')
model = ffn.get_model()
frame_suggestion = pd.DataFrame(columns=db.columns) # its empty
dontcarevalue = 'any'
list_turns = []
list_denied_restaurants = []
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

state_list = list(messages.keys())

# TODO handle contact information, 
#       put any inside the frame_user_input,
#        refactor s4_suggest_restaurant handlung
def state_transition(state,user_message = None):

    def load_suggestions():
        global frame_suggestion 
        for key, value in frame_user_input.items():
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

    def is_suggestions_empty():
        return frame_suggestion.empty
        
    def current_turn():
        return list_turns[-1]
    
    def is_current_state(condition_state : str):
        assert state in state_list
        return state == condition_state
    
    def is_current_prediction(dialog_act_no : int):
        return dialog_act[current_turn['dialog_act']] == dialog_act_no
    
    def predict_act(sentence : str):
        print((ffn.predict(model, sentence)[0]))
        return dialog_act[(ffn.predict(model, sentence)[0])]
        
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
        message = messages[state]
        print(message)
        return message

    def turn(system_message= None):
        if system_message:
            print(system_message)
        else:
            system_message = print_message()
        user_message = input()
        turn_frame = {"system_message": system_message, "user_message":user_message,
                       'dialog_act_system': predict_act(system_message),'dialog_act_user': predict_act(user_message),
                       "turn_index": len(list_turns)}
        if turn_frame['dialog_act_user'] == "inform":
            preference = get_preference(user_message)
            add_to_user_frame(preference)
        print(turn_frame)
        list_turns.append(turn_frame)
        
        
    
     
    # gets the first suggestion from frame and DELETE it from Frame 
    def pop_suggestion():
        global frame_suggestion
        suggestion = frame_suggestion.iloc[0]
        frame_suggestion = frame_suggestion.drop([suggestion.name])
        return  suggestion  
            

    

    def get_next_state():
        print(state)
        if is_current_state('s0_welcome'):
            turn()
            return 's1_ask_price'
            
        elif is_current_state('s1_ask_price'):
            if not is_pricerange_expressed():
                turn()
                return 's1_ask_price'
            return 's2_ask_area'
            
        elif is_current_state('s2_ask_area'):
            if not is_area_expressed():
                turn()
                return 's2_ask_area'
            return 's3_ask_food'
        
        elif is_current_state('s3_ask_food'):
            if not is_food_expressed():
                turn()
                return 's3_ask_food'
            return 's4_suggest_restaurant'
        
        # TODO : make this pretty
        elif is_current_state('s4_suggest_restaurant'):
            global frame_suggestion
            if is_suggestions_empty(): 
                load_suggestions()
            if not is_suggestions_empty():
                found_suggestion = False
                while not is_suggestions_empty():
                    suggestion = pop_suggestion()
                    if  not suggestion.restaurantname in list_denied_restaurants:
                        
                        found_suggestion = True
                        break
                if found_suggestion:
                    turn(f'how about this restaurant: {suggestion.restaurantname}')
                    
                
                    # get clasification
                    if current_turn()["dialog_act_user"] == "affirm":
                        return 's5_give_info'
                    elif current_turn()["dialog_act_user"] == 'reqalts':
                        list_denied_restaurants.append(suggestion.restaurantname)
                        return 's4_suggest_restaurant'
                else:
                    return "s7_restart"

               
            else:
                return "s7_restart"
            
        elif is_current_state('s7_restart'):
            turn(f'didnt found any with{frame_user_input}, start over')
            
            return "s1_ask_price"
    
        elif is_current_state('s5_give_info'):
            print("Which information do you want: Phone number, ")

        return 's6_bye'
    
    if is_current_state('s6_bye'):
        return 
    
    # load_suggestions()
    # #print_message()    
   # user_message = input()
    
    next_state = get_next_state()
    state_transition(next_state)
    
    
if __name__ == '__main__':
    
    state = state_transition('s0_welcome')
    print(state)

