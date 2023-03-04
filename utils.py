import string
import re
import torch
import spacy

from fastapi import HTTPException
from transformers import DistilBertModel
from gensim.utils import tokenize

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def load_stopwords(path: str):
    with  open(path, "r") as f:
        content = f.read()
        stop_words = content.split(',')
        f.close()
    return stop_words

STOPWORDS = load_stopwords('resources/stop_words.txt') 

class SnowballStemmer:
    def __init__(self, language='english'):
        self.language = language
        self.vowels = ['a', 'e', 'i', 'o', 'u']

    def stem(self, word):
        if self.language == 'english':
            if len(word) < 2:
                return word

            if word[-2:] == 'ss':
                return word

            # Step 1
            if word[-3:] == 'ied' or word[-3:] == 'ies':
                word = word[:-3] + 'i'
            elif word[-2:] == 'ed' or word[-2:] == 'es':
                if word[-4:-2] == 'at' or word[-4:-2] == 'bl' or word[-4:-2] == 'iz':
                    word = word[:-2]
                elif word[-3] == word[-4]:
                    word = word[:-2]
                elif word[-2] in self.vowels:
                    word = word[:-2] + 'e'

            # Step 2
            if word[-2:] == 'ic':
                word = word[:-2]
            elif word[-4:] == 'ance' or word[-4:] == 'ence' or word[-4:] == 'able' or word[-4:] == 'ible' or word[-3:] == 'ant' or word[-3:] == 'ent' or word[-3:] == 'ate' or word[-3:] == 'ive' or word[-3:] == 'ize':
                if len(word[:-4]) >= 2:
                    word = word[:-4]
            elif word[-5:] == 'ement':
                if len(word[:-5]) >= 2:
                    word = word[:-5]
            elif word[-3:] == 'ant' or word[-3:] == 'ent' or word[-3:] == 'ism':
                if len(word[:-3]) >= 2:
                    word = word[:-3]
            elif word[-4:] == 'ment' or word[-4:] == 'ness' or word[-4:] == 'less':
                if len(word[:-4]) >= 2:
                    word = word[:-4]

            # Step 3
            if word[-5:] == 'ement':
                if len(word[:-5]) >= 2 and (word[:-5][-1] == 'c' or word[:-5][-1] == 't'):
                    word = word[:-5]
            elif word[-4:] == 'ance' or word[-4:] == 'ence' or word[-4:] == 'able' or word[-4:] == 'ible' or word[-4:] == 'ment' or word[-4:] == 'less':
                if len(word[:-4]) >= 2:
                    word = word[:-4]
            elif word[-3:] == 'ate' or word[-3:] == 'iti' or word[-3:] == 'ous' or word[-3:] == 'ive' or word[-3:] == 'ize':
                if len(word[:-3]) >= 2:
                    word = word[:-3]

            # Step 4
            if word[-2:] == 'al' or word[-4:] == 'ance' or word[-4:] == 'ence' or word[-4:] == 'eror' or word[-4:] == 'able' or word[-4:] == 'ible' or word[-4:] == 'ment' or word[-5:] == 'ement':
                if len(word[:-2]) >= 2 and word[:-2][-1] == 'a':
                    word = word[:-2]
            
            # Step 5a
            if word[-4:] == 'al':
                if len(word[:-4]) >= 2:
                    word = word[:-4]
            elif word[-6:] == 'icalli':
                if len(word[:-6]) >= 2:
                    word = word[:-6] + 'ic'
            elif word[-4:] == 'ance' or word[-4:] == 'ence' or word[-4:] == 'able' or word[-4:] == 'ible':
                if len(word[:-4]) >= 2:
                    word = word[:-4]
            elif word[-4:] == 'ence' or word[-4:] == 'able' or word[-4:] == 'ible' or word[-4:] == 'ment':
                if len(word[:-4]) >= 2:
                    word = word[:-4]

            # Step 5b
            if word[-3:] == 'ant' or word[-3:] == 'ent' or word[-3:] == 'ism' or word[-3:] == 'ate' or word[-3:] == 'iti' or word[-3:] == 'ous' or word[-3:] == 'ive' or word[-3:] == 'ize':
                if len(word[:-3]) >= 2 and word[-4] == 'a':
                    word = word[:-3]

            # Step 5c
            if word[-1:] == 'e':
                if len(word[:-1]) >= 2:
                    if len(word[:-1]) == 2:
                        if word[0] not in self.vowels:
                            word = word[:-1]
                    else:
                        word = word[:-1]

            # Step 6
            if word[-2:] == 'al':
                if len(word[:-2]) >= 2:
                    word = word[:-2]

            return word

def porter_stemmer(word):
    """
    Apply the Porter stemming algorithm to a word.
    """
    # Define regular expressions for matching various patterns
    vowels = '[aeiou]'
    consonants = '[^aeiou]'
    m = f'^{consonants}?{vowels}[^aeiouwxy]$'
    m_eq_1 = f'^{consonants}?({vowels}[^aeiouwxy]|[aeiouy][^{vowels}])$'
    m_gt_1 = f'^{consonants}?({vowels}[^aeiouwxy]|[aeiouy][^{vowels}]).+$'

    def measure(word):
        return len(re.findall(f'{consonants}{vowels}', word))

    def ends_cvc(word):
        return re.search(m, word) is not None and re.search(m_eq_1, word) is None

    # Step 1a
    if word.endswith('sses'):
        word = word[:-2]
    elif word.endswith('ies'):
        word = word[:-2] + 'i'
    elif word.endswith('s'):
        word = word[:-1]

    # Step 1b
    if re.search('eed$', word):
        stem = re.sub('eed$', '', word)
        if len(stem) > 1:
            word = stem + 'ee'
    elif re.search('(ed|ing)$', word):
        stem = re.sub('(ed|ing)$', '', word)
        if re.search('[aeiouy]', stem):
            word = stem
            if re.search('at$', word):
                word = word + 'e'
            elif re.search('bl$', word):
                word = word + 'e'
            elif re.search('iz$', word):
                word = word + 'e'
            elif re.search('([^aeiouylsz])\1$', word):
                word = re.sub('([^aeiouylsz])\1$', '\\1', word)
            elif re.search('^([^aeiouy][aeiouy][^aeiouywxY])|(^[aeiouy][^aeiouywxY])$', word):
                word = word + 'e'

    # Step 1c
    if re.search('y$', word):
        stem = re.sub('y$', '', word)
        if re.search('[aeiouy]', stem):
            word = stem + 'i'

    # Step 2
    if re.search('(ational|tional|enci|anci|izer|bli|alli|entli|eli|ousli|ization|ation|ator|alism|iveness|fulness|ousness|aliti|iviti|biliti)$', word):
        stem = re.sub('(ational|tional|enci|anci|izer|bli|alli|entli|eli|ousli|ization|ation|ator|alism|iveness|fulness|ousness|aliti|iviti|biliti)$', '', word)
        if len(stem) > 1:
            word = stem

    # Step 3
    if re.search('(icate|ative|alize|iciti|ical|ful|ness)$', word):
        stem = re.sub('(icate|ative|alize|iciti|ical|ful|ness)$', '', word)
        if len(stem) > 1:
            word = stem

    # Step 4
    if re.search('al$', word):
        stem = re.sub('al$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ance$', word):
        stem = re.sub('ance$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ence$', word):
        stem = re.sub('ence$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('er$', word):
        stem = re.sub('er$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ic$', word):
        stem = re.sub('ic$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('able$', word):
        stem = re.sub('able$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ible$', word):
        stem = re.sub('ible$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ant$', word):
        stem = re.sub('ant$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ement$', word):
        stem = re.sub('ement$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ment$', word):
        stem = re.sub('ment$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ent$', word):
        stem = re.sub('ent$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('sion$', word):
        stem = re.sub('sion$', 's', word)
        if len(stem) > 1:
            word = stem
    elif re.search('tion$', word):
        stem = re.sub('tion$', 't', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ou$', word):
        stem = re.sub('ou$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ism$', word):
        stem = re.sub('ism$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ate$', word):
        stem = re.sub('ate$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('iti$', word):
        stem = re.sub('iti$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ous$', word):
        stem = re.sub('ous$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ive$', word):
        stem = re.sub('ive$', '', word)
        if len(stem) > 1:
            word = stem
    elif re.search('ize$', word):
        stem = re.sub('ize$', '', word)
        if len(stem) > 1:
            word = stem

    # Step 5: Remove a final 'e' (except for -le and -me)
    if re.search('e$', word):
        stem = re.sub('e$', '', word)
        if (measure(stem) > 1) or ((measure(stem) == 1) and (not ends_cvc(stem))):
            word = stem

    return word

# Remove website links
def remove_links(text):
    template = re.compile(r'https?://\S+|www\.\S+') 
    text = template.sub(r'', text)
    return text

# Remove HTML tags
def remove_html(text):
    template = re.compile(r'<[^>]*>') 
    text = template.sub(r'', text)
    return text

def text2words(text):
      return tokenize(text)
    
def to_lowercase(text):
    return text.lower()

# Remove stopwords
def remove_stopwords(words, stop_words):
    return [word for word in words if word not in stop_words]

# Remove none ascii characters
def remove_non_ascii(text):
    template = re.compile(r'[^\x00-\x7E]+') 
    text = template.sub(r'', text)
    return text

# Replace none printable characters
def remove_non_printable(text):
    template = re.compile(r'[\x00-\x0F]+') 
    text = template.sub(r' ', text)
    return text

# Remove special characters
def remove_special_chars(text):
        text = re.sub("'s", '', text)
        template = re.compile('["#$%&\'()\*\+-/:;<=>@\[\]\\\\^_`{|}~]') 
        text = template.sub(r' ', text)
        return text

# Replace multiple punctuation 
def replace_multiplt_punc(text):
        text = re.sub('[.!?]{2,}', '.', text)
        text = re.sub(',+', ',', text) 
        return text

def remove_punctuation(text):
    """Remove punctuation from list of tokenized words"""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# Remove numbers
def remove_numbers(text):
        text = re.sub('\d+', ' ', text)
        return text

def handle_spaces(text):
    # Remove extra spaces
    text = re.sub('\s+', ' ', text)
    
    # Remove spaces at the beginning and at the end of string
    text = text.strip()
    return text

def stem_words(words):
    """Stem words in text"""
    stemmer = porter_stemmer()
    return [stemmer.stem(word) for word in words]

lemmatizer = spacy.load('en_core_web_sm')
def lemmatize_words(words):
    """Lemmatize words in text"""
    lem_words = lemmatizer(' '.join(words))

    return [token.lemma_ for token in lem_words]

def clean_text(text, stemmer = 'lm'):
    
    #text = remove_pattern(text)
    text = remove_links(text)
    text = remove_html(text)
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_non_printable(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = text.lower()
    text = handle_spaces(text)
    
    words = text2words(text)
    words = remove_stopwords(words, STOPWORDS)
    #words = [word for word in words if word not in stop_words]
    
    # stem words using Porter stemmer or Snowball Stemmer algorithm
    
    stemmed_words = []
    if stemmer=='porter':
        for word in words:
            stemmed_words.append(porter_stemmer(word)) #Porter stemmer
            
    elif stemmer=='snowball':
        sb = SnowballStemmer(language='english')
        for word in words:
            stemmed_words.append(sb.stem(word)) #Snowball Stemmer
            
    elif stemmer=='lemmatize':
        stemmed_words = lemmatize_words(words) #Lemmatize text

    else:
        raise HTTPException(status_code=400, detail="Error invalid stemmer. Try 'porter' or 'snowball' or 'lemmatize'")
        
    return stemmed_words

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 6)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        out = hidden_state[:, 0]
        out = self.classifier(out)
        return out
    
def predict_text(text, model, tokenizer, device=DEVICE):
    encoded_text = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=320,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True
    )
    # Tokenize the text and convert to input IDs
    input_ids = torch.tensor(encoded_text['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoded_text['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(encoded_text['token_type_ids'], dtype=torch.long).unsqueeze(0).to(device)

    # Generate the attention mask
    attention_mask = (input_ids != 0).float()

    # Make the prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    # Convert the logits to probabilities
    probs = torch.sigmoid(outputs)

    # Convert the probabilities to binary predictions
    preds = probs.detach().cpu().numpy()

    # Convert the binary predictions to class labels
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    results = {label: round(pred.item(), 4) for label, pred in zip(labels, preds[0])}
    return results