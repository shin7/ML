from pickle import load
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from utils import generate_desc, load_set, load_clean_descriptions, load_doc, load_photo_features
from constants import TEST_SET, MODEL_FILE, TOKEN, MAX_LENGTH, DESCRIPTION, FEATURE

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# load the tokenizer
tokenizer = load(open(TOKEN, 'rb'))
# pre-define the max sequence length (from training)
max_length = int(load_doc(MAX_LENGTH))
print('Description Length: %d' % max_length)

# prepare test set

# load test set
test = load_set(TEST_SET)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions(DESCRIPTION, test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features(FEATURE, test)
print('Photos: test=%d' % len(test_features))

# load the model
model = load_model(MODEL_FILE)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)