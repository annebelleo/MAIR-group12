# MAIR-group12

figs/1b_dialogdiagram.drawio.png

run dialog_manager.py

example output:
(uu_courses) (base) robin@Ideapad:~/uu/mair_chat/MAIR-group12$ /home/robin/anaconda3/envs/uu_courses/bin/python /home/robin/uu/mair_chat/MAIR-group12/src/dialog_manager.py
s0_welcome
Welcome to the chatbot
Hi!
{'system_message': 'Welcome to the chatbot', 'user_message': 'Hi!', 'dialog_act_system': 'null', 'dialog_act_user': 'hello', 'turn_index': 0}
s1_ask_price
What is your budget?
any
{'system_message': 'What is your budget?', 'user_message': 'any', 'dialog_act_system': 'request', 'dialog_act_user': 'inform', 'turn_index': 1}
{'area': None, 'food': None, 'pricerange': 'any'}
s1_ask_price
s2_ask_area
Which area do you prefer?
center
{'system_message': 'Which area do you prefer?', 'user_message': 'center', 'dialog_act_system': 'request', 'dialog_act_user': 'inform', 'turn_index': 2}
{'area': 'centre', 'food': None, 'pricerange': 'any'}
s2_ask_area
s3_ask_food
What type of food do you like?
italian
{'system_message': 'What type of food do you like?', 'user_message': 'italian', 'dialog_act_system': 'request', 'dialog_act_user': 'inform', 'turn_index': 3}
{'area': 'centre', 'food': 'italian', 'pricerange': 'any'}
s3_ask_food
s4_suggest_restaurant
I have found caffe uno. It is an expensive restaurant in the centre part of town that serves italian food.                    
Are you interested in it?
no
{'system_message': 'I have found caffe uno. It is an expensive restaurant in the centre part of town that serves italian food.                    \nAre you interested in it?', 'user_message': 'no', 'dialog_act_system': 'inform', 'dialog_act_user': 'negate', 'turn_index': 4}
s4_suggest_restaurant
I have found pizza hut city centre. It is an cheap restaurant in the centre part of town that serves italian food.                    
Are you interested in it?
yes
{'system_message': 'I have found pizza hut city centre. It is an cheap restaurant in the centre part of town that serves italian food.                    \nAre you interested in it?', 'user_message': 'yes', 'dialog_act_system': 'inform', 'dialog_act_user': 'affirm', 'turn_index': 5}
s5_give_info
What information do you want to know?
number
{'system_message': 'What information do you want to know?', 'user_message': 'number', 'dialog_act_system': 'null', 'dialog_act_user': 'request', 'turn_index': 6}
Here is the['phone']: ('01223 323737',). Do you need more information?
no   
{'system_message': "Here is the['phone']: ('01223 323737',). Do you need more information?", 'user_message': 'no ', 'dialog_act_system': 'request', 'dialog_act_user': 'negate', 'turn_index': 7}
Good bye
None
