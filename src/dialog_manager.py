
import pandas as pd
from preference_extraction import get_preference, request_extraction, consequent_extraction
from suggestion_manager import Suggestion_Manager
import models.feed_forward as ffn
import models.random_forest as rf

import tokenizer_manual as tok
import numpy as np
import reasoner
import logging
import time
import system_messages
import socket
import datetime
import json


# TODO: reasoner conf["language"]
# TODO: Bug reask for additional requirements



# extra feature configuration
conf = {
    "is_ask_additional_requirement" : True, # (Ask user for additional requirements, such as romantic)
    "is_ask_levenstein_correction" : False, # (Ask user about correctness of match for Levenshtein results)
    "answer_delay" : 0, # Introduce a delay before showing system responses (in seconds)
    "is_output_caps" : False, #OUTPUT IN ALL CAPS OR NOT
    "is_direct_search" : True ,# Start offering suggestions 
                            # after the first preference type is recognized vs. wait until all preference types are recognized
                            # BUG: if direct search is enabled, the system will not ask for additional requirements 
    "is_show_debug_information" : False, # Show debug information
    "language" : "GENZ" # "GENZ" or "FORMAL

}



log_frames_level = 10

# if you want information about frames: log_frames_level < log_frames_level
# else log_frames_level >= log_frames_level
logging_level = 0
if conf["is_show_debug_information"]:
    logging_level = 9
else:
    logging_level = 11
logging.getLogger().setLevel(logging_level)             


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
                            's7_restart',
                            's8_suggestion_available']
        self.state = self.state_list[0]
        self.tokenizer = tok.load_tokenizer("res/models/tokenizer_0.pkl")
        self.model = rf.load_model("res/models/random_forest_0.pkl")
        self.suggestion_manager = Suggestion_Manager()
        self.turn_index = 0
        self.frame_current_turn = None
        time_stamp = str(datetime.datetime.now()).replace(":", "-").replace(" ", "_")
        self.user_data_frame_json = {
            "device_name": socket.gethostname(),
            "time_stamp": time_stamp,
            "configuration": conf,
            "number_suggestions": 0,
            "number_restarts": 0,
            "turns" : []
        }
        self.frame_user_input = {"area": None,
                            "food": None,
                            "pricerange": None}
        
    def save_meta_frame(self):
        with open(f'res/user_data/{self.user_data_frame_json["device_name"]}_{self.user_data_frame_json["time_stamp"]}.json', 'w') as f:
            json.dump(self.user_data_frame_json, f, indent=4)
            
        

    # clears user frame for restarting
    def clear_frame_user(self):
        self.frame_user_input = {"area": None,
                    "food": None,
                    "pricerange": None}

    
    # check if current state is the same as the condition state
    def is_current_state(self,condition_state : str):
        assert self.state in self.state_list
        return self.state == condition_state
    
    # return the dialog act of a sentence 
    def predict_act(self,sentence : str):
        embedded_sentence = self.tokenizer(pd.Series([sentence]))
        return dialog_act[self.model.predict((embedded_sentence))[0]]
    
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
        if not self.is_current_state('s0_welcome'):
            time.sleep(conf["answer_delay"])
        if conf["is_output_caps"]:
            print(system_message.upper())
        else:
            print(system_message.lower())
        time_before_input = datetime.datetime.now()
        user_message = input()


        response_time =datetime.datetime.now() -  time_before_input  
        response_time = str(response_time)
        response_time = response_time.split(":")
        
        seconds = (float(response_time[0]) * 60 + float(response_time[1])) * 60 + float(response_time[2])

        turn_frame = {"system_message": system_message, "user_message":user_message,
                    'dialog_act_system': self.predict_act(system_message),'dialog_act_user': self.predict_act(user_message),
                    "turn_index": self.turn_index, "response_seconds": seconds}
        logging.log(log_frames_level,turn_frame)
        self.turn_index += 1
        self.user_data_frame_json["turns"].append(turn_frame)
        self.frame_current_turn = turn_frame
        self.save_meta_frame()
       
    # makes a turn to ask the user for information about a category (area, food or pricerange)
    # first checks if user dialog act is a inform, otherwise we dont to do anything 
    # if the user input is not clear, the system will ask again (user message is unclear if levenstein is used)
    # if the user input is clear, the system will add the preference to the user frame
    def ask_for_inform(self,category = None, message = None):
        self.turn(message)
        if self.frame_current_turn['dialog_act_user'] == "inform":
            preference = get_preference(self.frame_current_turn["user_message"], category)
            if len(preference) == 0:
                self.ask_for_inform(message=system_messages.MESSAGES["re_ask_inform"][conf["language"]] % self.frame_current_turn['user_message'], category=category)
                return
            is_used_leven = not list(preference.values())[0] in self.frame_current_turn["user_message"].split()
            if conf["is_ask_levenstein_correction"] and is_used_leven:
                self.turn(system_messages.MESSAGES["re_inform"][conf["language"]] % list(preference.values())[0])
                if self.frame_current_turn["dialog_act_user"] == "affirm":
                    self.add_to_user_frame(preference)
                    logging.log(log_frames_level,self.frame_user_input)

            else:                    
                self.add_to_user_frame(preference)
                logging.log(log_frames_level,self.frame_user_input)

    # ask for addional requirements like (romantic or touristic...)
    # if no additional requirements are given, we just move on
    # otherwise filter in the suggestion manager for the additional requirement
    def ask_additional_requierments(self):
        self.turn(system_messages.MESSAGES["ask_add_requirements"][conf["language"]])
        if not self.frame_current_turn["dialog_act_user"] == "negate":
            additional_req = consequent_extraction(self.frame_current_turn["user_message"])
            logging.log(log_frames_level,additional_req)
            if len(additional_req) > 0:
                self.suggestion_manager.filter(additional_req[0]) 
                return additional_req[0]
                
            else:
                return self.ask_additional_requierments()
            
    # suggestion a restaurant
    # 1 we load the suggestions from the csv file with the user frame
    # 2 if there is more than one suggestion we ask for additional requirements
    # 3 we propose a suggestion, if there is no suggestion available we restart the conversation
    def suggest_restaurant(self):

        self.suggestion_manager.load_suggestions(self.frame_user_input,
                                        path = 'res/restaurant_extra_info.csv', is_user_frame_complete= not conf["is_direct_search"])
        additional_req = None
        if self.suggestion_manager.get_number_suggestions() > 1 and conf["is_ask_additional_requirement"]:
            additional_req = self.ask_additional_requierments()
        logging.log(log_frames_level,f'Suggestion available: {not self.suggestion_manager.is_suggestions_exhausted()}')
        self.suggestion_manager.propose_suggestion()
        if self.suggestion_manager.is_suggestions_exhausted():
            self.suggestion_manager.reset_suggestions()
            self.state = 's7_restart'
        else:
            self.user_data_frame_json["number_suggestions"] += 1
            suggestion_data = self.suggestion_manager.get_suggestion_information(["restaurantname","pricerange","area","food"])
            suggestion_message = system_messages.MESSAGES["suggest_restaurant"][conf["language"]] % suggestion_data
            if additional_req:
                suggestion_message = suggestion_message + reasoner.get_reasoning(additional_req, conf["language"])
            suggestion_message = suggestion_message + system_messages.MESSAGES["suggest_interess"][conf["language"]]
            self.turn(suggestion_message)
            # get clasification
            if self.frame_current_turn["dialog_act_user"] == "affirm":
                self.state = 's5_give_info'
            elif self.frame_current_turn["dialog_act_user"] == "request":
                self.state = 's5_give_info'
                self.give_contact_information()
            elif self.frame_current_turn["dialog_act_user"] == 'reqalts' or self.frame_current_turn["dialog_act_user"] == 'negate':
                self.state = 's4_suggest_restaurant'
 
    
    
    # first make a turn to ask which information the user wants,
    # extract the information from the user message and make a turn with the information   
    def give_contact_information(self):
        if self.frame_current_turn["dialog_act_user"] == "request":
            contact_information = request_extraction(self.frame_current_turn["user_message"])
            if len(contact_information) > 0:
                data = self.suggestion_manager.get_suggestion_information(contact_information)
                self.turn( system_messages.MESSAGES["give_contact"][conf["language"]] % (contact_information, data))
                self.give_contact_information()
        elif self.frame_current_turn["dialog_act_user"] == "negate":
            self.state = 's6_bye'
        else:
            self.turn(system_messages.MESSAGES["re_give_contact"][conf["language"]])
            self.give_contact_information()
        
    # load suggestions and check if there is exactly one suggestion available
    # if there are more than one suggestion available we reset the suggestions
    def is_suggestion_available(self):
        self.suggestion_manager.load_suggestions(self.frame_user_input,
                                        path = 'res/restaurant_extra_info.csv', is_user_frame_complete =not conf["is_direct_search"])
        if not self.suggestion_manager.get_number_suggestions() == 1:
                self.suggestion_manager.reset_suggestions()
        return not self.suggestion_manager.is_suggestions_exhausted()
       
    # process the current state
    # this followes the diagram
    #   
    def process_states(self):

        
        logging.log(log_frames_level,self.state)
        if self.is_current_state('s0_welcome'):
            self.ask_for_inform(message= system_messages.MESSAGES["welcome"][conf["language"]])
            self.state =  's1_ask_price'
        

        elif self.is_current_state('s1_ask_price'):
            if conf["is_direct_search"] and self.is_suggestion_available():
                self.state = 's4_suggest_restaurant'
            elif not self.is_pricerange_expressed():
                self.ask_for_inform("pricerange", message= system_messages.MESSAGES["ask_budget"][conf["language"]])
                self.state = 's1_ask_price'
            else:
                self.state = 's2_ask_area'
            
        elif self.is_current_state('s2_ask_area'):
            if conf["is_direct_search"] and self.is_suggestion_available():
                self.state = 's4_suggest_restaurant'
            elif not self.is_area_expressed():
                self.ask_for_inform("area", message= system_messages.MESSAGES["ask_area"][conf["language"]])
                self.state =  's2_ask_area'
            else:
                self.state =  's3_ask_food'
        
        elif self.is_current_state('s3_ask_food'):
            if conf["is_direct_search"] and self.is_suggestion_available():
                self.state = 's4_suggest_restaurant'
            elif not self.is_food_expressed():
                self.ask_for_inform("food", message= system_messages.MESSAGES["ask_food"][conf["language"]])
                self.state = 's3_ask_food'
            else:
                self.state = 's4_suggest_restaurant'
        
        elif self.is_current_state('s4_suggest_restaurant'):
            self.suggest_restaurant()
           
        elif self.is_current_state('s5_give_info'):
            self.turn(system_messages.MESSAGES["ask_contact"][conf["language"]])
            self.give_contact_information()
        elif self.is_current_state('s7_restart'):
            self.clear_frame_user()
            self.user_data_frame_json["number_restart"] += 1
            print(system_messages.MESSAGES["restart"][conf["language"]])
            self.state = "s1_ask_price"
        else: 
            self.state =  's6_bye'
        
        if self.is_current_state('s6_bye'):
            self.turn(system_messages.MESSAGES["bye"][conf["language"]])    
        else:
            self.process_states()
            
    
if __name__ == '__main__':
    diaglog_manager = Dialog_Manager()
    diaglog_manager.process_states()
    
