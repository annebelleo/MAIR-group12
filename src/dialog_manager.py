from data_preparation import class2descript
import preference_extraction
frame_schema = ["restaurantname","pricerange","area","food","phone","addr","postcode"]
frame =  dict(zip(frame_schema, [None]*len(frame_schema)))

print(class2descript)

messages = {
    's0_welcome' : {'message' : 'Welcome to the chatbot'},
    's1_ask_area' : {'message' : 'Which area?'},
    's2_suggest_restaurant' : {'message' : 'Eat at restaurant x'},
    'sn_terminate' : {'message' : 'thanks for using dialog'},
    'sE_invalid_input' : {'message' : 'error'}
}


def state_transition(state, userInput = None):
    
    def add_preferences(preferences : dict):
        '''
        updates preferences in frame
        '''
        for key in preferences.keys():
            frame['key'] = preferences[key] # modify frame
            
    def get_statemsg(state : str):
        messages[state]['message']
        return
    
    def clear_frame():
        frame =  dict(zip(frame_schema, [None]*len(frame_schema)))
        
    def is_area_expressed():
        return frame['area'] == None
        
    
    # Predict userInput
    prediction = 6 # Hardcoded 'inform'
    invalid_input = 'sE_invalid_input'
    
    if state == 's0_welcome':
        # Options that are valid in state, and
        if prediction == 6:
            # extract preference
            preferences = {'area' : 'west'} # hardcoded
            add_preferences(preferences)            
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
    elif state == 's1_ask_area':
        if prediction == 6:
            return 0
    else:
        return 0
    
if __name__ == '__main__':
    state = state_transition('s0_welcome')
    print(state)

