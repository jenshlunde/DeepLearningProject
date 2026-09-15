import kagglehub                            #pip install kagglehub
from datasets import load_dataset           #pip install datasets
from huggingface_hub import hf_hub_download #pip install --upgrade huggingface_hub

#HUGGING FACE data
#HF auth login
### TERMINAL COMMANDS NEEDED FOR HUGGING FACE DATASET ACCESS ###
# hf skills add
# hf auth login
################################################################

# # Login using e.g. `huggingface-cli login` to access this dataset
#ds_twitch = load_dataset("lparkourer10/twitch_chat")                                            #94.8 MB
#ds_discord = load_dataset("SaisExperiments/Discord-Unveiled-Compressed")                        #118 GB
ds_OLID = load_dataset("christophsonntag/OLID")                                                 #11.4 MB

#KAGGLE DATA
# Download latest version
#path = kagglehub.dataset_download("jef1056/discord-data")                                       #22.7 GB
path = kagglehub.dataset_download("julian3833/jigsaw-toxic-comment-classification-challenge")   #140 MB

print("Path to dataset files:", path)