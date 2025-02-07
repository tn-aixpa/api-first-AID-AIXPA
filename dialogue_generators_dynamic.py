import os
from tools import chunker, dialogue, retrieval, openai_gpt, span, prompt_composer
import json
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from typing import Callable, Dict, List, Union

with open('keys.json') as keys_file:
    keys = json.load(keys_file)

with open('methods_dynamic_dialogue/prompts.json') as keys_file:
    prompts = json.load(keys_file)



def get_dynamic_generation_options():
    generation_options = [
        {
            "generation_method": "aixpa_dynamic",
            "description": "genera dialoghi per supportare operatori comunali",
            "endpoint": "/dialogue_generation_dynamic/",
            "roles": [
                {
                    "label": "speaker_1",
                    "name": "Operator",
                    "ground": False
                },
                {
                    "label": "speaker_2",
                    "name": "ChatBot",
                    "ground": True
                }
            ]
        }
    ]
    return generation_options


def generate_dialogue_dynamic(generation_opton, *args, **kwargs):
    try:
        # Get the function object from the global namespace
        func = globals()[generation_opton]
        # Call the function with the provided arguments and return the result
        return func(*args, **kwargs)
    except KeyError:
        return f"Generation option {generation_opton} not found."
    except TypeError as e:
        return str(e)



def aixpa_dynamic(documents_list: List[str], dialogue_list: List[Dict[str, str]], speaker: str, options_number: int, manual_selected_grounds: List[str]):
    """
    TODO
    """
    
    roles =  {
        "speaker_1": "Operatore", 
        "speaker_2": "Chatbot"
    }
    
    ground_required_dict =  {
        "speaker_1": False,
        "speaker_2": True
    } 
    
    chunks = chunker.Chunker_llama_index(
        documents_list = documents_list, 
        chunk_size     = 300,
        chunk_overlap  = 50
    )

    generator = openai_gpt.GenerationModelChat(
        model_name          = "gpt-4o-mini-2024-07-18",  
        question_sys_prompt = "question_prompt",
        access_token        = keys['OPENAI_API_AIXPA']
    )
    
    dial = dialogue.Dialogue(turns = dialogue_list)
    
    retr = retrieval.Retriever_llamaindex_bm25(
        knowledge_base=chunks,
        name="BM25",
        top_k=options_number
    )

    if len(manual_selected_grounds) == 0:
        referenced_chunks: List[List[TextNode]] = [[chunk] for chunk in retr.retrieve(dial.get_last_turn())]
    else:
        # SimpleNamespace allows to call the variables with the dot notation
        referenced_chunks: List[List[TextNode]] = [[chunker.TextNode(text = el.strip(), metadata = None) for el in manual_selected_grounds]]
        #[TextNode(text = retrieved_doc.strip(), metadata = text_metadata_dict[retrieved_doc])
    # for node in retrieved_chunks:
    #     # assuming only one chunk per segment is retrieved by retriever
    #     print(node[0].metadata, node[0].text)
    # print("--########--")

    prompt_composer_obj = prompt_composer.PromptComposer(dial, "OpenAI")

    ##############################
    ###### MODEL GENERATION ######
    ##############################
    candidates_texts = list()

    ###### QUESTION ######
    if speaker == "speaker_1":

        if len(dial) != 0:
            # not the first question
            prompt_composer_obj.set_system_prompt(prompts["AIXPA"]["QUESTION"]["GENERIC_INFORMAL_QUEST_V1.2"])  
        else:
            # first question
            prompt_composer_obj.set_system_prompt(prompts["AIXPA"]["QUESTION"]["GENERIC_INFORMAL_QUEST_V1.2"])

        generated_raw_text = f"<root>{generator.generate_text(prompt_composer_obj.get_full_prompt())}</root>"
        try:
            candidate_txt = [question.text for question in ET.fromstring(generated_raw_text).findall('.//question')]
        except ParseError as pe:
            print(f"generated text cannot be parsed, returning empty question...\nGenerated text:\"{generated_raw_text}\"")
            candidate_txt = [""]

        for candidate in candidate_txt:
            candidates_texts.append(candidate)

    ###### ANSWER ######
    elif speaker == "speaker_2":

        for chunks in referenced_chunks:
            ground_text = "".join([f"<context>{chunk.text.strip()}</context>" for chunk in chunks])
            prompt_composer_obj.set_system_prompt(prompts["AIXPA"]["ANSWER"]["ANSW_COMUNI_V1"]+ "\n" + ground_text)
            
            generated_raw_text = f"<root>{generator.generate_text(prompt_composer_obj.get_full_prompt())}</root>"
            try:
                candidate_txt = [answer.text for answer in ET.fromstring(generated_raw_text).findall('.//answer')][0]
            except ParseError as pe:
                warnings.warn("generated text cannot be parsed, returning empty answer...\n", generated_raw_text)
                candidate_txt = ""
            
            candidates_texts.append(candidate_txt)

    else:
         raise ValueError(f"Unexpected speaker \"{speaker}\"")

    print("########")
    print(candidates_texts)
    print("########")

    ##########################
    ### OUTPUT PREPARATION ###
    ##########################
    future_turn_options = list()
    
    for candidate_index, candidate_txt in enumerate(candidates_texts):
        grounds_list = list()
        if ground_required_dict[speaker] and len(manual_selected_grounds) == 0:
            # NOTE assumes retr.retrieve only gets one value for each generation,
            # not considered cases in which text is generated over two or more grounds
            ground_info = dict()
            g_text = referenced_chunks[candidate_index][0].text
            g_doc  = referenced_chunks[candidate_index][0].metadata["document_id"]
            index_start, index_end = span.find_indexes(documents_list[g_doc], g_text)
            
            ground_info = {
                "text":         g_text,
                "file_index":   g_doc,
                "offset_start": index_start,
                "offset_end":   index_end,
            }
            grounds_list.append(ground_info)
        
        else:
            g_text      = ""
            g_doc       = ""
            index_start = 0
            index_end   = 0
            
        future_turn_options.append({
                "speaker":   speaker,
                "turn_text": candidate_txt,
                "ground":    grounds_list   
            }
        )
        print(future_turn_options)
    return future_turn_options