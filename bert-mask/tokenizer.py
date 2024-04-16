import pandas as pd
from transformers import BertTokenizer, LineByLineTextDataset
from tokenizers import BertWordPieceTokenizer


data_set_dir = "./data-set"
model_dir = "./saved_model"

df = pd.read_csv(f'{data_set_dir}/reviews.csv')
df.dropna(inplace=True)
mlm_df = df[['title', 'body']].copy()
# mlm_df.head()

# Write the sub dataframe into txt file
with open(f'{data_set_dir}/review_data.txt', 'w', encoding='utf-8') as f:
    for title, body in zip(mlm_df.title.values, mlm_df.body.values):
        f.write(title + '\n')
        f.write(body + '\n')
        

tokenizer = BertWordPieceTokenizer()
# vocab_size adalah jumlah vocab/kata yang diinginkan
tokenizer.train(files=f"{data_set_dir}/review_data.txt", vocab_size=30522)
tokenizer.save_model(f'{model_dir}/', 'phone_review')

# Read the vocabulary file
vocab_file_dir = f'{model_dir}/phone_review-vocab.txt'
custom_tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)
sentence = 'Motorola V860 is a good phone'
encoded_input = custom_tokenizer.tokenize(sentence)
print(encoded_input)

# Load BERT default tokenizer -> checkin result is not all tokenizer
# bert_default_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = 'Motorola V860 is a good phone'
# encoded_input = bert_default_tokenizer.tokenize(sentence)
# print(encoded_input)
# result -> ['motorola', '##60', 'is', 'a', 'good', 'phone']

# Convert input text data to tokens for custom bert model
dataset= LineByLineTextDataset(
    tokenizer = custom_tokenizer,
    file_path = 'data/review_data.txt',
    block_size = 128
)
 
print('No. of lines: ', len(dataset))