from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# See how many tokens are in the vocabulary
print(tokenizer.vocab_size)

# Tokenize the sentence
tokens = tokenizer.tokenize("I heart Generative AI")

# Print the tokens
print(tokens)

# Show the token ids assigned to each token
print(tokenizer.convert_tokens_to_ids(tokens))