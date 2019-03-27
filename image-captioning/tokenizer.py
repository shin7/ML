from pickle import dump
from utils import load_set, load_clean_descriptions, create_tokenizer, max_length
from constants import TRAIN_SET, DESCRIPTION, MAX_LENGTH, TOKEN

# save max_length to file
def save_max_length(max_length, filename):
	file = open(filename, 'w')
	file.write(max_length)
	file.close()

# load training dataset (6K)
train = load_set(TRAIN_SET)
print('Dataset: %d' % len(train))

# descriptions
train_descriptions = load_clean_descriptions(DESCRIPTION, train)
print('Descriptions: train=%d' % len(train_descriptions))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# save max_length
save_max_length(str(max_length), MAX_LENGTH)

# save the tokenizer
dump(tokenizer, open(TOKEN, 'wb'))