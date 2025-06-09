import os
from tools import chunker, dialogue, retrieval, openai_gpt, span
import json
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from lorax import Client
from openai import OpenAI
# Global model and tokenizer variable, this will hold the loaded model once it's initialized
model = None
tokenizer = None


from huggingface_hub import hf_hub_download

# Set model repository and filename
repo_id = "Stefano-M/aixpa_amicifamiglia_short_prompt" 
file_name = "adapter_model.safetensors"  
local_folder = "adapters/aixpa"

# Ensure local directory exists
os.makedirs(local_folder, exist_ok=True)

# Download adapter
local_path = hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=local_folder)
print(f"Adapter downloaded to: {local_path}")
file_name = "adapter_config.json"  # Example file to download
local_path = hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=local_folder)
print(f"Adapter downloaded to: {local_path}")
file_name = "README.md"
local_path = hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=local_folder)
print(f"Adapter downloaded to: {local_path}")

def get_openai_chat_format(turns):

    if turns and turns[0]['speaker'] == "speaker_2":
        turns.pop(0)
    
    openai_chat = []

    if len(turns)> 0:
        for turn in turns:
            
            turn_dict = dict()

            if turn["speaker"] == "speaker_1":
                turn_dict["role"] = "user" 
                turn_dict["content"] = turn["turn_text"]
                openai_chat.append(turn_dict)

            if turn["speaker"] == "speaker_2":
                turn_dict["role"] = "assistant" 
                turn_dict["content"] = turn["turn_text"]
                openai_chat.append(turn_dict)          

    return(openai_chat)


# Load keys and prompts
with open('keys.json') as keys_file:
    keys = json.load(keys_file)

with open('methods_dynamic_dialogue/prompts.json') as prompts_file:
    prompts = json.load(prompts_file)

filename = "conversationlog.txt"

if not os.path.exists(filename):
    with open(filename, "w") as file:
        pass  # Creates the file
    print(f"{filename} created.")
else:
    print(f"{filename} already exists.")

def write_list_to_file(data_list, filename="conversationlog.txt"):
    with open(filename, "a") as file:  # Open in append mode
        # for item in data_list:
        #     file.write(str(item) + "\n")
        from datetime import date

        today = date.today()
        file.write(str(today)+" "+str(data_list)+"\n")
        file.close()
    print(f"Data written to {filename}.")

def load_model_and_tokenizer(hf_token):
    
    global model, tokenizer

    # Check if the model and tokenizer are already loaded, if not, load them once
    if model is None or tokenizer is None:
        print("Loading model and tokenizer...")

        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        # adapter_path = "Stefano-M/aixpa_amicifamiglia_short_prompt"
        adapter_path = "adapters/aixpa"

        # Set up 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

        # Load the base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",  # Automatically map layers to available devices (e.g., GPU/CPU)
            use_auth_token=hf_token
        )

        # Wrap the model with PEFT to load the adapter
        model = PeftModel.from_pretrained(model, adapter_path)
        
        print("Loaded PEFT adapter config:")
        print(model.peft_config)
        

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

        print("Available adapters:", model.peft_config.keys())
        print("Active adapter:", model.active_adapter)
    
        
    return model, tokenizer

def generate_dialogue_dynamic(generation_opton, *args, **kwargs):
    try:
        func = globals()[generation_opton]
        return func(*args, **kwargs)
    except KeyError:
        return f"Generation option {generation_opton} not found."
    except TypeError as e:
        return str(e)

def aixpa_chatbot_local(documents_list, dialogue_list, user, language, lorax_client, hf_token):
    
    print("Using HF token:", hf_token)
    
    dialogue_list_extended = dialogue_list.copy()
    write_list_to_file(dialogue_list_extended)
    
    # Load model and tokenizer (if not already loaded)
    model, tokenizer = load_model_and_tokenizer(hf_token)

    # Speaker roles in the conversation
    roles = {
        "speaker_1": "user",
        "speaker_2": "assistant",
        "user":      "speaker_1",
        "assistant": "speaker_2"
    }
    
    # Number of chunks to retrieve for grounding references
    options_number = 1

    # Chunk your knowledge base
    chunks = chunker.Chunker_llama_index(
        documents_list  = documents_list, 
        chunk_size      = 150,
        chunk_overlap   = 50
    )

    # Build a Dialogue object (if needed for your usage)
    dial = dialogue.Dialogue(turns=dialogue_list)

    # Initialize a retriever for later retrieval
    retr = retrieval.Retriever_llamaindex_bm25(
        knowledge_base=chunks,
        name="BM25",
        top_k=options_number
    )

    prompt_template = '''You are an helpful assistant from the public administration, use a <<STYLE>> tone.
    The user is a <<ROLE>>.
    Your task is to provide a relevant answer to the user using the provided evidence and past dialogue history.
    The evidence is contained in <document> tags.Be proactive asking for the information needed to help the user.
    When you receive a question, answer by referring exclusively to the content of the document.
    Answer in Italian.
    <document><<DOCUMENTS>></document>'''

    # Replace placeholders
    if user == "cittadino":
        prompt_template = prompt_template.replace("<<ROLE>>", "Citizen")
    elif user == "operatore":
        prompt_template = prompt_template.replace("<<ROLE>>", "Public Administration Worker")
    
    if language == "informale":
        prompt_template = prompt_template.replace("<<STYLE>>", "informal")
    else:
        prompt_template = prompt_template.replace("<<STYLE>>", "formal")

    prompt_template = prompt_template.replace("<<DOCUMENTS>>", "\n".join(documents_list))

    # Prepare the chat history for "system"/"user"/"assistant" roles
    chat_dict = [{"role": "system", "content": prompt_template}]

    # If the dialogue starts with speaker_2, remove it (avoid a leading assistant turn)
    if dialogue_list and dialogue_list[0]['speaker'] == "speaker_2":
        dialogue_list.pop(0)

    # Include user/assistant turns in chat_dict (except the very last turn if you want to treat it as the user query)
    for turn in dialogue_list:
        chat_dict.append({"role": roles[turn['speaker']], "content": turn['turn_text']})

    # Now convert this chat structure into a single prompt string.
    prompt_str = tokenizer.apply_chat_template(chat_dict, tokenize=False)

    # print(prompt_str)
    # print("this was the prompt")
    print("prompt received")

    # Generate the next turn from the prompt string
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,  # Limit the number of tokens in the response
        do_sample=True,  # Enable sampling for variability
        top_k=40,        # Top-k sampling to control diversity
        temperature=0.8  # Sampling temperature to control randomness
    )

    # Decode and print the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = response.split("assistant")[-1]

    print("--------------------")
    print("------RESPONSE------")
    print("--------------------")
    print(response)


    # Build your return dict
    reply = response
    return_dict = {}
    return_dict["message"] = reply

    dialogue_list_extended.append({'speaker': 'speaker_2', 'turn_text': reply})
    write_list_to_file(dialogue_list_extended)

    # Retrieve grounding references from the newly generated reply
    chunks_for_grounding = retr.retrieve(reply)
    grounds_list = [c.text for c in chunks_for_grounding]
    return_dict["ground"] = grounds_list

    # print("Reply:\n", reply)
    # print("Grounding:", grounds_list)
    return return_dict


def aixpa_chatbot_kubeai (documents_list, dialogue_list, user, language, kubeai_client, hf_token):
    
    dialogue_list_extended = dialogue_list.copy()
    write_list_to_file(dialogue_list_extended)

    speaker = "speaker_2"

    roles = {
        "speaker_1": "user",
        "speaker_2": "assistant",
        "user": "speaker_1",
        "assistant": "speaker_2"
        } 
    ground_required_dict =  {
        "speaker_1": False,
        "speaker_2": True
        } 
    
    options_number = 3
    
    chunks = chunker.Chunker_llama_index(
            documents_list  = documents_list, 
            chunk_size    = 150,
            chunk_overlap = 50
            )    

    dial= dialogue.Dialogue(turns = dialogue_list)
    
 
    retr = retrieval.Retriever_llamaindex_bm25(
        knowledge_base=chunks,
        name="BM25",
        top_k=options_number
        )

    model_name =  "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    prompt = '''You are an helpful assistant from the public administration, use a <<STYLE>> tone.
    The user is a <<ROLE>>.
    Your task is to provide a relevant answer to the user using the provided evidence and past dialogue history.
    The evidence is contained in <document> tags.Be proactive asking for the information needed to help the user.
    When you receive a question, answer by referring exclusively to the content of the document.
    Answer in Italian.
    <document><<DOCUMENTS>></document>'''



    if user == "cittadino":
        prompt = prompt.replace("<<ROLE>>", "Citizen")

    if user == "operatore":
        prompt = prompt.replace("<<ROLE>>", "Public Operator")

    if language == "informale":
        prompt = prompt.replace("<<STYLE>>", "informal")

    if language == "formale":
        prompt = prompt.replace("<<STYLE>>", "formal")
    
    prompt = prompt.replace("<<DOCUMENTS>>", "\n".join(documents_list))

    openaidict_sys = dict()
    openaidict_user = dict()

    openaidict_sys["role"] = "system"
    openaidict_sys["content"] = prompt
    openaidict_user = get_openai_chat_format(dialogue_list)
    # openaidict_user["role"] = "user"
    # openaidict_user["content"] = dialogue_list[1]["turn_text"]
    # chat_dict = [{"role": "system", "content": prompt}]
    # if dialogue_list[0]['speaker'] == "speaker_2":
    #     dialogue_list.pop(0)
    openailist = []
    openailist.append(openaidict_sys)
    openailist.extend(openaidict_user)
    print(openailist)
    # for turn in dialogue_list[0:-1]:
    # for turn in dialogue_list:
    #     chat_dict.append({"role": roles[turn['speaker']], "content": turn['turn_text']})
        
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    # inputs = tokenizer.apply_chat_template(chat_dict, tokenize=False)



    client = OpenAI(
        base_url = 'http://kubeai.kubeai.svc.cluster.local/openai/v1',
        api_key='ollama', # required, but unused
    )
    message = client.chat.completions.create(
        model="amici-famiglia-2",
        messages=openailist,
        temperature=0.3,
        # max_completion_tokens=1000
    ).choices[0].message.content

    print(message)

    return_dict = dict()
    # message_parts = message.split("\n\n")
    
    # message_parts.pop(0)
    # message = "\n\n".join(message_parts)
    return_dict["message"] = message

    
    dialogue_list_extended.append({'speaker': 'speaker_2', 'turn_text': message})
    write_list_to_file(dialogue_list_extended)

    # return_dict = dict() #DA CANCELLARE
    # message = "Servono politiche per la promozione del benessere familiare" #DA CANCELLARE
    # return_dict["message"] = "Servono politiche per la promozione del benessere familiare" #DA CANCELLARE

    chunks = retr.retrieve(message)
    grounds_list = []
    for c in chunks:
        grounds_list.append(c.text)
    return_dict["ground"] = grounds_list
    # print(return_dict)
    return(return_dict)


def aixpa_chatbot_lorax (documents_list, dialogue_list, user, language, lorax_client, hf_token):
    
    dialogue_list_extended = dialogue_list.copy()
    write_list_to_file(dialogue_list_extended)
    
    speaker = "speaker_2"

    roles = {
        "speaker_1": "user",
        "speaker_2": "assistant",
        "user": "speaker_1",
        "assistant": "speaker_2"
        } 
    ground_required_dict =  {
        "speaker_1": False,
        "speaker_2": True
        } 
    
    options_number = 3
    
    chunks = chunker.Chunker_llama_index(
            documents_list  = documents_list, 
            chunk_size    = 150,
            chunk_overlap = 50
            )    

    dial= dialogue.Dialogue(turns = dialogue_list)
    
 
    retr = retrieval.Retriever_llamaindex_bm25(
        knowledge_base=chunks,
        name="BM25",
        top_k=options_number
        )

    model_name =  "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    promt = '''You are an helpful assistant from the public administration, use a <<STYLE>>.
    The user is a <<ROLE>>.
    Your task is to provide a relevant answer to the user using the provided evidence and past dialogue history.
    The evidence is contained in <document> tags.\nBe proactive asking for the information needed to help the user.
    When you receive a question, answer by referring exclusively to the content of the document.
    Answer in Italian'''


    if user == "cittadino":
        prompt = prompt.replace("<<ROLE>>", "Citizen")

    if user == "operatore":
        prompt = prompt.replace("<<ROLE>>", "Public Operator")

    if language == "informale":
        prompt = prompt.replace("<<STYLE>>", "informal")

    if language == "formale":
        prompt = prompt.replace("<<STYLE>>", "formal")
    
    prompt = prompt.replace("<<DOCUMENTS>>", "\n".join(documents_list))

    chat_dict = [{"role": "system", "content": prompt}]
    if dialogue_list[0]['speaker'] == "speaker_2":
        dialogue_list.pop(0)
    
    # for turn in dialogue_list[0:-1]:
    for turn in dialogue_list:
        chat_dict.append({"role": roles[turn['speaker']], "content": turn['turn_text']})
        
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    inputs = tokenizer.apply_chat_template(chat_dict, tokenize=False)

    client = Client(base_url=lorax_client)

    message = client.generate(
                prompt=inputs,
                adapter_id="run_merged_new_prompt_",
                adapter_source="local",
                max_new_tokens=5000,
                temperature=0.1,
                top_p=0.99,
                return_full_text=False,
                ).generated_text

    print(message)

    return_dict = dict()
    message_parts = message.split("\n\n")
    
    message_parts.pop(0)
    message = "\n\n".join(message_parts)
    return_dict["message"] = message

    
    dialogue_list_extended.append({'speaker': 'speaker_2', 'turn_text': message})
    write_list_to_file(dialogue_list_extended)

    # return_dict = dict() #DA CANCELLARE
    # message = "Servono politiche per la promozione del benessere familiare" #DA CANCELLARE
    # return_dict["message"] = "Servono politiche per la promozione del benessere familiare" #DA CANCELLARE

    chunks = retr.retrieve(message)
    grounds_list = []
    for c in chunks:
        grounds_list.append(c.text)
    return_dict["ground"] = grounds_list
    # print(return_dict)
    return(return_dict)



def aixpa_chatbot_test (documents_list, dialogue_list, user, language, lorax_client, hf_token):
    
    speaker = "speaker_2"

    roles = {
        "speaker_1": "user",
        "speaker_2": "assistant",
        "user": "speaker_1",
        "assistant": "speaker_2"
        } 
    ground_required_dict =  {
        "speaker_1": False,
        "speaker_2": True
        } 
    
    options_number = 3
    
    
    chunks = chunker.Chunker_llama_index(
            documents_list  = documents_list, 
            chunk_size    = 150,
            chunk_overlap = 50
            )    

    
 
    retr = retrieval.Retriever_llamaindex_bm25(
        knowledge_base=chunks,
        name="BM25",
        top_k=options_number
        )


    premade_chat = ["Potresti elencarmi alcune azioni che posso inserire nel macroambito Misure economiche?",
    "Volentieri, ecco alcuni esempi:\n- Introduzione di un contributo per i pannolini lavabili;\n- Previsione di agevolazioni tariffarie e contributi per servizi educativi e di conciliazione, come corsi estivi o tagesmutter;\n- Introduzione di un voucher sportivo per le famiglie;\n- Introduzione di agevolazioni tariffarie e contributi per attivitа ricreative/culturali/aggregative/formative, ad esempio sottoscrivendo un accordo con la Provincia Autonoma di Trento per attivare una scontistica presso una struttura del tuo comune per i possessori dell'Euregio Family Pass.\nSei interessato ad approfondire una di queste azioni? Ti posso aiutare a scriverne il testo per poi inserirla nel tuo Piano comunale.",
    "In cosa consiste il Voucher Sportivo?",
    "Il voucher sportivo è una agevolazione per aiutare le famiglie in difficoltà economica nonché le famiglie numerose alla partecipazione dei figli nelle attività sportive.\n\nSe ti può sembrare interessante, posso aiutarti a scrivere un'azione intorno al voucher sportivo.",
    "No grazie, non è quello che pensavo. Piuttosto, che cosa è il family pass?",
    "l’EuregioFamilyPass è la carta famiglia della Provincia autonoma di Trento, gratuita e valida per tutte le famiglie residenti con almeno un figlio minorenne, a prescindere dal reddito. Il possesso della card permette alle famiglie di usufruire di sconti e tariffe speciali presso numerosi enti convenzionati, sia pubblici che privati, in Trentino, Alto Adige e Tirolo.\nLa lista è agilmente consultabile sulla mappa interattiva. In Trentino le convenzioni spaziano tra musei, mezzi di trasporto, agriturismi, ostelli, ristoranti, piscine, impianti di risalita, teatro per bambini, assicurazioni sanitarie integrative, e sono in continua crescita.\nSei interessato ad aderire al progetto con una struttura del tuo comune?",
    "Ottimo! Mi aiuti a scrivere l'azione per uno sconto sul circolo comunale di tennis?",
    "Volentieri! Mi sapresti dare un po' di informazioni? Avrei bisogno di conoscere l'obiettivo dell'azione, chi sono i destinatari, dove si trova la struttura, che tipo di sconto pensi di introdurre ed in generale ogni informazione utile a meglio descrivere l'azione.",
    "Vogliamo dare un'agevolazione alle famiglie che hanno la card di cui parlavamo prima oltreché inventivare l'uso dei campi da tennis. Il tutto per raggiungere uno stilo di vita salutare. Riserviamo questa agevolazione a chi possiede la card.\nL'agevolazione riguarda il circolo tennis di via degli Alpini. Nel circolo ci sono 2 campi da tennis e uno da calcetto, oltre a spogliatoi e bar. Il circolo è attivo dalle 9 alle 22 da aprile a settembre.\nChi possiede la card ha diritto al 10% di sconto",
    "Perfetto. Ecco la proposta di testo per l'azione del tuo piano:\nTITOLO: Adesione all'EuregioFamilyPass con il circolo tennis\nTASSONOMIA: Agevolazioni tariffarie e contributi attivitа ricreative/culturali/aggregative/formative\nMACRO-AMBITO: Misure economiche\nOBIETTIVO: Fornire un'agevolazione alle famiglie in possesso dell'EuregioFamilyPass e promuovere l'utilizzo dei campi da tennis, anche al fine di diffondere stili di vina sani.\nDESCRIZIONE: Il progetto prevede la stipulazione di un accordo con la Provincia Autonoma di Trento per inserire il locale circolo tennis nella lista di strutture che riconoscono uno sconto ai clienti in possesso dell'EuregioFamilyPass.\nIl circolo tennis è ubicato in via degli Alpini, comprende 2 campi da tennis e uno di calcetto, è attrezzato con spogliatoi e bar e lavora con orario 9-22 dal mese di aprile al mese di settembre. \nL'intenzione è quella di introdurre in questa struttura uno sconto del 10% per i possessori  dell'EuregioFamilyPass.\n\nFammi sapere se questo testo ti può andare bene, altrimenti ti posso aiutare a migliorarlo.",
    "Grazie, potresti provare a darmi un'altra versione del testo? Deciderò dopo quale usare. ",
    "Certamente, ecco qui una versione alternativa del testo con una nuova descrizione: \n\nTITOLO: Adesione all'EuregioFamilyPass con il circolo tennis\nTASSONOMIA: Agevolazioni tariffarie e contributi attivitа ricreative/culturali/aggregative/formative\nMACRO-AMBITO: Misure economiche\nOBIETTIVO: Offrire un'agevolazione alle famiglie in possesso dell'EuregioFamilyPass e incentivare l'utilizzo dei campi da tennis, promuovendo stili di vita sani.\nDESCRIZIONE: Il progetto prevede la stipulazione di un accordo con la Provincia Autonoma di Trento per includere il circolo tennis locale tra le strutture che aderiscono uno sconto ai titolari dell'EuregioFamilyPass.\nIl circolo, situato in via degli Alpini, dispone di due campi da tennis e un campo da calcetto, ed è dotato di spogliatoi e bar. L'orario di apertura va dalle 9:00 alle 22:00, da aprile a settembre.\nI possessori dell'EuregioFamilyPassL'obiettivo avranno un 10% di sconto.\n\nRimango a dispositizione se hai bisogno di un'aiuto per altro.",
    ]

    
    message = premade_chat[len(dialogue_list)-1]

    return_dict = dict()

    return_dict["message"] = message

    chunks = retr.retrieve(message)
    grounds_list = []
    for c in chunks:
        grounds_list.append(c.text)
    return_dict["ground"] = grounds_list
    # print(return_dict)
    return(return_dict)


