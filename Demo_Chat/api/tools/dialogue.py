import json

# def dialogue_to_string(dialogue_list):
#     dialogue_string = ""
#     for turn in dialogue_list:
#          dialogue_string = dialogue_string + turn["speaker"] + ": " + turn["turn_text"] + "\n"
#     return(dialogue_string)


class Dialogue:
     def __init__(self, 
               turns = []):
        self.turns = turns
        self.turn_numbers = len(self.turns)
        self.dialogue_string= self.dialogue_to_string()
        self.openai_chat = self.get_openai_chat_format()

     def dialogue_to_string(self):
          dialogue_string = ""
          for turn in self.turns:
               dialogue_string = dialogue_string + turn["speaker"] + ": " + turn["turn_text"] + "\n"
          return(dialogue_string)
          
     def get_last_turn(self):
          try:
               return(self.turns[-1]["turn_text"])
          except:
               return("")
     def get_last_speaker(self):
          try:
               return(self.turns[-1]["speaker"])
          except:
               return("")
          
     def get_openai_chat_format(self):
          openai_chat = []

          if len(self.turns)> 0:
               for turn in self.turns:
                    
                    turn_dict = dict()

                    if turn["speaker"] == "speaker_1":
                         turn_dict["role"] = "user" 
                         turn_dict["content"] = turn["turn_text"]
                         openai_chat.append(turn_dict)

                    if turn["speaker"] == "speaker_2":
                         turn_dict["role"] = "assistant", 
                         turn_dict["content"] = turn["turn_text"]
                         openai_chat.append(turn_dict)          

          return(openai_chat)

# class GenerationModel:
#     def __init__(self,
#                  model_name = DEFAULT_OPENAI_MODEL,
#                  temperature = DEFAULT_TEMPERATURE,
#                  max_tokens = DEFAULT_MAX_TOKENS,
#                  access_token = "",
#                  question_sys_prompt = ""):

#         self.model_name = model_name
#         self.access_token = access_token
#         self.model = self.set_model()
#         self.temperature = temperature
#         self.max_tokens = max_tokens
#         self.question_sys_prompt = question_sys_prompt 

