Notes for project:

possibly data source:
Unlabeled(?), Twitch_Chat, Twitch, 94.8 MB: https://huggingface.co/datasets/lparkourer10/twitch_chat
Unlabeled, Discord, 118 GB (might need a lot of cleaning): https://huggingface.co/datasets/SaisExperiments/Discord-Unveiled-Compressed
Unlabeled chat, Discord, 22.7 GB: https://www.kaggle.com/datasets/jef1056/discord-data
Labeled toxic data, Wikipedia talk-page comments, 140 MB: https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge/data?select=sample_submission.csv
Labeled Toxic Data, Twitter, 11.4 MB: https://huggingface.co/datasets/christophsonntag/OLID 

Article: 
"Toxicity in Twitch Chats: An LLM-Based Analysis Across Gaming Communities" - https://arxiv.org/pdf/2605.24000
"TinyBERT: Distilling BERT for Natural Language Understanding" - https://arxiv.org/abs/1909.10351
"Discord Unveiled: A Comprehensive Dataset of Public Communication (2015-2024)" - https://arxiv.org/abs/2502.00627 

Project angles:
Transfer learning for other platforms - Discord, chatrooms, ..
TinyML for NLP - 
Problem about different languages



Performance metrics: Accuracy, latency, memory cost, computational cost, ...?
- Related metrics: #params, 
Metrics that should stay fixed over testing(?): Tokenizer, platform it runs on, optimizer,
Things to change: data_size (32float vs 16int vs 8int), wide v. narrow models, bottlenecked (?)
Things to look out for: 'camouflaged' text - shit / sh1t - fuck / f_u_c_k
