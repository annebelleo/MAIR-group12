from data_preparation import class2descript
import pandas as pd
from preference_extraction import get_preference, request_extraction, consequent_extraction
from suggestion_manager import Suggestion_Manager
#import preference_extraction
import models.feed_forward as ffn
import numpy as np
import reasoner


is_ask_levenstein = True

                


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
class Dialog_Manager():
    def __init__(self):
        self.state_list =[  's0_welcome' ,
                            's1_ask_price',
                            's2_ask_area',
                            's3_ask_food',
                            's4_suggest_restaurant',
                            's5_give_info',
                            's6_bye',
                            's7_restart']
        self.state = self.state_list[0]
        self.model = ffn.get_model()
        self.suggestions = Suggestion_Manager()
        self.list_turns = []

        self.frame_user_input = {"area": None,
                            "food": None,
                            "pricerange": None}


    

    def current_turn(self):
            return self.list_turns[-1]
        
    def is_current_state(self,condition_state : str):
        assert self.state in self.state_list
        return self.state == condition_state
    
    
    def predict_act(self,sentence : str):
        return dialog_act[(ffn.predict(self.model, sentence)[0])]
        
    def add_to_user_frame(self,preferences : dict):
        '''
        updates preferences in frame
        '''
        for key in preferences.keys():
            self.frame_user_input[key] = preferences[key] # modify frame
    
    
    def is_area_expressed(self):
        return not self.frame_user_input['area'] == None

    def is_food_expressed(self):
        return not self.frame_user_input['food'] == None

    def is_pricerange_expressed(self):
        return not self.frame_user_input['pricerange'] == None
    
    def turn(self,system_message):
        print(system_message)
        user_message = input()
        turn_frame = {"system_message": system_message, "user_message":user_message,
                    'dialog_act_system': self.predict_act(system_message),'dialog_act_user': self.predict_act(user_message),
                    "turn_index": len(self.list_turns)}
        print(turn_frame)
        self.list_turns.append(turn_frame)
        
    def ask_for_inform(self,category = None, message = None):
        self.turn(message)
        if self.current_turn()['dialog_act_user'] == "inform":
                preference = get_preference(self.current_turn()["user_message"], category)
                if len(preference) == 0:
                    self.ask_for_inform(message=f"I didn't understand: {self.current_turn()['user_message']}")
                is_used_leven = not list(preference.values())[0] in self.current_turn()["user_message"].split()
                if is_ask_levenstein and is_used_leven:
                    self.turn(f"did you mean {list(preference.values())[0]}?")
                    if self.current_turn()["dialog_act_user"] == "affirm":
                        self.add_to_user_frame(preference)
                        print(self.frame_user_input)
                    else: 
                        self.ask_for_inform(message=f"what {list(preference.keys())[0]} did you mean?")
                else:                    
                    self.add_to_user_frame(preference)
                    print(self.frame_user_input)
                
            
    def get_next_state(self):
        print(self.state)
        if self.is_current_state('s0_welcome'):
            self.ask_for_inform(message= "Hi how can I help you?")
            return 's1_ask_price'
            
        elif self.is_current_state('s1_ask_price'):
            if not self.is_pricerange_expressed():
                self.ask_for_inform("pricerange", message= "What is your budget?")
                return 's1_ask_price'
            return 's2_ask_area'
            
        elif self.is_current_state('s2_ask_area'):
            if not self.is_area_expressed():
                self.ask_for_inform("area", message= "Which area you want to go?")
                return 's2_ask_area'
            return 's3_ask_food'
        
        elif self.is_current_state('s3_ask_food'):
            if not self.is_food_expressed():
                self.ask_for_inform("food", message= "What type of food do you want?")
                return 's3_ask_food'
            return 's4_suggest_restaurant'
        
        # TODO : make this pretty
        elif self.is_current_state('s4_suggest_restaurant'):
            
            self.suggestions.load_suggestions(self.frame_user_input,
                                        path = 'res/restaurant_extra_info.csv')
            if self.suggestions.get_number_suggestions() > 1:
                self.turn("Do you have additional requirements?")
                if not self.current_turn()["dialog_act_user"] == "negate":
                    preference = consequent_extraction(self.current_turn()["user_message"])
                    print(preference)
                    if len(preference) > 0:
                        self.suggestions.filter(preference[0]) 
                    else:
                        return 's4_suggest_restaurant'
            

            self.suggestions.propose_suggestion()
            if self.suggestions.is_suggestions_exhausted():
                self.suggestions.reset_suggestions()
                return 's7_restart'
            
            else:
                # NOTE: Example of querying the data from the current suggestion
                suggestion_data = self.suggestions.get_suggestion_information(["restaurantname","pricerange","area","food"])
                self.turn("I have found %s. It is an %s restaurant in the %s part of town that serves %s food.\
                    \nAre you interested in it?" % suggestion_data)
                    
    
                # get clasification
                if self.current_turn()["dialog_act_user"] == "affirm":
                    return 's5_give_info'
                elif self.current_turn()["dialog_act_user"] == 'reqalts' or self.current_turn()["dialog_act_user"] == 'negate':
                    #list_denied_restaurants.append(suggestion.restaurantname)
                    return 's4_suggest_restaurant'

        
        elif self.is_current_state('s5_give_info'):
            self.turn("What information do you want to know?")
            if self.current_turn()["dialog_act_user"] == "request":
                contact_information = request_extraction(self.current_turn()["user_message"])
                data = self.suggestions.get_suggestion_information(contact_information)
                self.turn(f"Here is the{contact_information}: {data}. Do you need more information?")
                if self.current_turn()["dialog_act_user"] == "request":
                    return "s5_give_info"
                else:
                    return "s6_bye"
            else:
                return "s6_bye"
            
        elif self.is_current_state('s7_restart'):
            self.turn(f'didnt found any with{self.frame_user_input}, start over')
            
            return "s1_ask_price"

        return 's6_bye'
    
    
    def state_transition(self):
            
        self.state = self.get_next_state()
        self.state_transition()
        if self.is_current_state('s6_bye'):
            print("Good bye")
            return 

    
    
if __name__ == '__main__':
    diaglog_manager = Dialog_Manager()
    diaglog_manager.state_transition()
    
