from data_preparation import class2descript
import pandas as pd
#import preference_extraction
db = pd.read_csv('res/restaurant_info.csv')

frame_suggestion = pd.DataFrame(columns=db.columns) # its empty
dontcarevalue = 'any'

frame_user_input = {"area": None,
                    "food": None,
                    "pricerange": None}
                
messages = {
    's0_welcome' : 'Welcome to the chatbot',
    's1_ask_area' : 'Which area?',
    's2_suggest_restaurant' : 'Eat at restaurant %s',
    's3': '',
    's4' : '',
    's5' : '',
    's6' : '',
    's7' : '',
    's8' : '',
    's9' : '',
    'st_terminate' : 'thanks for using dialog',
    'sE_invalid_input' : 'error'
}

state_list = list(messages.keys())


def state_transition(state, userInput = None):
    def load_suggestions():
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
            frame_suggestion = db.copy()
        else:
            frame_suggestion = db.query(query)   
        frame_suggestion.dropna(axis = 0, inplace = True)   
        
    load_suggestions() 

    def current_state(condition_state : str):
        assert state in state_list
        return state == condition_state
    
    def current_prediction(dialog_act_no : int):
        return prediction == dialog_act_no
    
    def predict_act(sentence : str):
        # Not implemented yet. Use external library
        return 6
        
    def add_frame_to(preferences : dict):
        '''
        updates preferences in frame
        '''
        for key in preferences.keys():
            frame_user_input['key'] = preferences[key] # modify frame
            
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
        return frame_user_input['area'] == None
        
    
    # Predict userInput
    prediction = predict_act(userInput)
    invalid_input = 'sE_invalid_input'
    
    if current_state('s0_welcome'):
        # Options that are valid in state, and
        if current_prediction(6):
            # extract preference
            preferences = {'area' : 'west'} # hardcoded
            add_frame_to(preferences)            
        else: # No valid input provided
            nextState = state
            print(get_statemsg(invalid_input))
            return nextState
        
        if is_area_expressed():
            nextState = 's1_ask_area'
            print(messages[nextState]['message'])
        else:
            nextState = 's2_suggest_restaurant'
            print(messages[nextState]['message'])
            
        return nextState
    
    elif current_state('s1_ask_area'):
        if prediction == 6:
            return 0
        
    elif current_state('s2'):
        pass
    
    elif current_state('s3'):
        pass
    
    elif current_state('s4'):
        pass
    
    elif current_state('s5'):
        pass
    
    elif current_state('s6'):
        pass
    
    elif current_state('s7'):
        pass
    
    elif current_state('s8'):
        pass
    
    elif current_state('s9'):
        pass
    
    else:
        return 0
    
if __name__ == '__main__':
    state = state_transition('s0_welcome')
    print(state)

