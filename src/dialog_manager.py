from data_preparation import class2descript
import pandas as pd
from preference_extraction import get_preference, request_extraction, consequent_extraction
from suggestion_manager import Suggestion_Manager
#import preference_extraction
import models.feed_forward as ffn
import numpy as np
import reasoner
import logging
# test 1, tomas
# test 2 model problem
# test 3, not in chart


is_ask_levenstein = True
logging_level = 10
logging.getLogger().setLevel(11)             


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
        self.suggestion_manager = Suggestion_Manager()
        self.list_turns = []

        self.frame_user_input = {"area": None,
                            "food": None,
                            "pricerange": None}


    
    # returns current turn of conversation
    # each turn is a message from the system and the user 
    def get_current_turn(self):
        return self.list_turns[-1]
    
    # check if current state is the same as the condition state
    def is_current_state(self,condition_state : str):
        assert self.state in self.state_list
        return self.state == condition_state
    
    # return the dialog act of a sentence 
    def predict_act(self,sentence : str):
        return dialog_act[(ffn.predict(self.model, sentence)[0])]
    
    # add preference to user frame
    def add_to_user_frame(self,preferences : dict):
        '''
        updates preferences in frame
        '''
        for key in preferences.keys():
            self.frame_user_input[key] = preferences[key] # modify frame
    
    # checks if area is already in the user frame
    def is_area_expressed(self):
        return not self.frame_user_input['area'] == None
    # checks if food is already in the user frame
    def is_food_expressed(self):
        return not self.frame_user_input['food'] == None
    # checks if the pricerange is already in the user frame
    def is_pricerange_expressed(self):
        return not self.frame_user_input['pricerange'] == None
    # make a turn:
    # 1. print system message
    # 2. ask for user input (user message)
    # 3. predict dialog act of system message and user message
    # 4. add turn_frame to list of turns
    def turn(self,system_message):
        print(system_message)
        user_message = input()
        turn_frame = {"system_message": system_message, "user_message":user_message,
                    'dialog_act_system': self.predict_act(system_message),'dialog_act_user': self.predict_act(user_message),
                    "turn_index": len(self.list_turns)}
        logging.log(logging_level,turn_frame)
        self.list_turns.append(turn_frame)
    
    # makes a turn to ask the user for information about a category (area, food or pricerange)
    # first checks if user dialog act is a inform, otherwise TODO what is happening??
    # if the user input is not clear, the system will ask again (user message is unclear if levenstein is used)
    # if the user input is clear, the system will add the preference to the user frame
    def ask_for_inform(self,category = None, message = None):
        self.turn(message)
        if self.get_current_turn()['dialog_act_user'] == "inform":
            preference = get_preference(self.get_current_turn()["user_message"], category)
            if len(preference) == 0:
                self.ask_for_inform(message=f"I didn't understand: {self.get_current_turn()['user_message']}")
            is_used_leven = not list(preference.values())[0] in self.get_current_turn()["user_message"].split()
            if is_ask_levenstein and is_used_leven:
                self.turn(f"did you mean {list(preference.values())[0]}?")
                if self.get_current_turn()["dialog_act_user"] == "affirm":
                    self.add_to_user_frame(preference)
                    logging.log(logging_level,self.frame_user_input)
                else: 
                    self.ask_for_inform(message=f"what {list(preference.keys())[0]} did you mean?")
            else:                    
                self.add_to_user_frame(preference)
                logging.log(logging_level,self.frame_user_input)

    # ask for addional requirements like (romantic or touristic...)
    # if no additional requirements are given, we just move on
    # otherwise filter in the suggestion manager for the additional requirement
    def ask_additional_requierments(self):
        self.turn("Do you have additional requirements?")
        if not self.get_current_turn()["dialog_act_user"] == "negate":
            additional_req = consequent_extraction(self.get_current_turn()["user_message"])
            logging.log(logging_level,additional_req)
            if len(additional_req) > 0:
                self.suggestion_manager.filter(additional_req[0]) 
                return additional_req[0]
            else:
                self.ask_additional_requierments()
            
    # suggestion a restaurant
    # 1 we load the suggestions from the csv file with the user frame
    # 2 if there is more than one suggestion we ask for additional requirements
    # 3 we propose a suggestion, if there is no suggestion available we restart the conversation
    def suggest_restaurant(self):

        self.suggestion_manager.load_suggestions(self.frame_user_input,
                                        path = 'res/restaurant_extra_info.csv')
        additional_req = None
        if self.suggestion_manager.get_number_suggestions() > 1:
            additional_req = self.ask_additional_requierments()

        self.suggestion_manager.propose_suggestion()
        if self.suggestion_manager.is_suggestions_exhausted():
            self.suggestion_manager.reset_suggestions()
            self.state = 's7_restart'
        else:
            suggestion_data = self.suggestion_manager.get_suggestion_information(["restaurantname","pricerange","area","food"])
            suggestion_message = "I have found %s. It is an %s restaurant in the %s part of town that serves %s food." % suggestion_data
            if additional_req:
                suggestion_message = suggestion_message + reasoner.get_reasoning(additional_req)
            suggestion_message = suggestion_message + "\nAre you interested in it?"
            self.turn(suggestion_message)
            # get clasification
            if self.get_current_turn()["dialog_act_user"] == "affirm":
                self.state = 's5_give_info'
            elif self.get_current_turn()["dialog_act_user"] == 'reqalts' or self.get_current_turn()["dialog_act_user"] == 'negate':
                #list_denied_restaurants.append(suggestion.restaurantname)
                self.state = 's4_suggest_restaurant'
    
    
    # first make a turn to ask which information the user wants,
    # extract the information from the user message and make a turn with the information   
    def ask_and_give_information(self):
        self.turn("What information do you want to know?")
        if self.get_current_turn()["dialog_act_user"] == "request":
            contact_information = request_extraction(self.get_current_turn()["user_message"])
            data = self.suggestion_manager.get_suggestion_information(contact_information)
            self.turn(f"Here is the{contact_information}: {data}. Do you need more information?")
            if self.get_current_turn()["dialog_act_user"] == "request":
                self.state = "s5_give_info"
                return
        self.state = "s6_bye"
            
    # process the current state
    # this followes the diagram
    #   
    def process_state(self):
        logging.log(logging_level,self.state)
        if self.is_current_state('s0_welcome'):
            self.ask_for_inform(message= "Hi how can I help you?")
            self.state =  's1_ask_price'
            
        elif self.is_current_state('s1_ask_price'):
            if not self.is_pricerange_expressed():
                self.ask_for_inform("pricerange", message= "What is your budget?")
                self.state = 's1_ask_price'
            self.state = 's2_ask_area'
            
        elif self.is_current_state('s2_ask_area'):
            if not self.is_area_expressed():
                self.ask_for_inform("area", message= "Which area you want to go?")
                self.state =  's2_ask_area'
            self.state =  's3_ask_food'
        
        elif self.is_current_state('s3_ask_food'):
            if not self.is_food_expressed():
                self.ask_for_inform("food", message= "What type of food do you want?")
                self.state = 's3_ask_food'
            self.state = 's4_suggest_restaurant'
        
        elif self.is_current_state('s4_suggest_restaurant'):
            self.suggest_restaurant()
           
        elif self.is_current_state('s5_give_info'):
            self.ask_and_give_information()
        elif self.is_current_state('s7_restart'):
            self.turn(f'didnt found any with{self.frame_user_input}, start over')
            
            self.state = "s1_ask_price"
        else: 
            self.state =  's6_bye'
        
        if not self.is_current_state('s6_bye'):
            self.process_state()    

    


    
    
if __name__ == '__main__':
    diaglog_manager = Dialog_Manager()
    diaglog_manager.process_state()
    
