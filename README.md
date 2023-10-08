# This is a chat bot for restaurant suggestions:
## sources (root/src):
* For data handling (like loading) check the data_preparation.py file.
* In the folder src/models/ you can find all of our models. To evaluate and use them, use model_eval.py. This file has two helper file model_testing_method.py and visualization.py
* tokenizer.py is a implementation for word embedding.
* to start a the chatbot run dialog_manager.py. This file uses several helpers
    * preference_extraction.py to get information out of the user message (using levenstein distance).
    * reasoner.py to get a reasoning for romatic, touristic,...
    * suggestion_manager.py to load suggestions out of the database.
## res (root/res):
Here you can find all ressources. You can find the database and the stored models here.
## figures (root/fig):
here you can find all figures and tables which are importent to the project.  
