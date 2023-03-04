import requests
import random
import string
url = "http://localhost:8000/predict"
def test_predict():
    text="This is a test comment."
    stemmers=["porter","snowball","lemmatize"]
    for stemmer in stemmers:
        data = {
            'text': text,
            'stemmer': stemmer
        }
        response = requests.post(url,json=data)
        assert response.status_code == 200
        assert isinstance(response.json(), dict)
        assert set(response.json()['predictions'].keys()) == {"toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"}

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def test_predict_too_long():
    text=get_random_string(2001)
    stemmers=["porter","snowball","lemmatizer"]
    for stemmer in stemmers:
        data = {
            'text': text,
            'stemmer': stemmer
        }
        response = requests.post(url,json=data)
        assert response.status_code == 400  
        assert response.json() == {"detail": "Text length exceeds 2000 characters"}
        