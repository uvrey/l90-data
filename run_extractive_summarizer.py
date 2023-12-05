import argparse
import json
import tqdm
from extractive_summarizer import ExtractiveSummarizer

# gets arguments from command line
args = argparse.ArgumentParser()
args.add_argument('--train_data', type=str, default='data/train.greedy_json.json')
args.add_argument('--eval_data', type=str, default='data/validation.json')
args.add_argument('--pred_data_dest', type=str, default='prediction_file.json')
args = args.parse_args()

# builds an extractive summarizer 
model = ExtractiveSummarizer()

# loads training data
with open(args.train_data, 'r') as f:
    train_data = json.load(f)

# pre-processes training data
train_articles = [article['article'] for article in train_data]
eval_article_ids = [article['id'] for article in train_data]
train_highlight_decisions = [article['greedy_n_best_indices'] for article in train_data]

preprocessed_train_articles = model.preprocess(train_articles, eval_article_ids)
model.train(preprocessed_train_articles, train_highlight_decisions)

# loads evaluation data
with open(args.eval_data, 'r') as f:
    eval_data = json.load(f)

# pre-processes evaluation data
eval_articles = [article['article'] for article in eval_data]
eval_article_ids = [article['id'] for article in eval_data]
preprocessed_eval_articles = model.preprocess(eval_articles, eval_article_ids)

# prepares summaries as a json
summaries = model.predict(preprocessed_eval_articles)

# prepares output summaries
eval_out_data = [{'article': article, 'summary': summary} for article, summary in zip(eval_articles, summaries)]

# writes file
with open(args.pred_data_dest , 'w', encoding='utf-8') as file:
    json.dump(eval_out_data, file, indent=4)

# notifies user
print(f'Summary built! View in `prediction_file.json` unless you\'ve changed this :)')