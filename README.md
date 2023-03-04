## FastAPI Rest App for Toxic Comment Classification

This is a FastAPI REST app for predicting whether a given text comment is toxic or not. It uses a pre-trained DistilBERT model and tokenizer to perform the prediction.

Check out this <a href="https://github.com/BrianMburu/Distiled-BERT-model-training-pytorch.git">Repo</a> to see how I trained the DistilBERT model.

### Setup

To run this app locally, follow these steps:

1. Clone the repository using git clone https://github.com/BrianMburu/Toxic-Comment-Classification-App.git.
2. Install the required packages by running pip install -r requirements.txt.
3. Run the app `using uvicorn main:app --reload` command and visit http://localhost:8000/docs in your web browser to interact with the API. You can also use curl as shown in the examples below.

### Usage

Once the app is up and running, visit http://localhost:8000/docs in your web browser to interact with the API. You can input a text comment and choose whether to apply stemming or lemmatization before predicting its toxicity.

### Input

- text: string - The input text comment to predict.
- stemmer: string - Either "snowball" or "lemmatize" to choose between applying snowball stemming or lemmatization respectively.

### Output

- predictions: dictionary - A dictionary with keys representing each of the six toxicity categories and their corresponding prediction score.
- Execution Time: float - The time taken to perform the prediction, in seconds.

### Examples

#### Input:

```json
{
  "text": "I don't think that's a good idea",
  "stemmer": "lemmatize"
}
```

#### Output:

```json
{
  "predictions": {
    "toxic": 0.7986,
    "severe_toxic": 0.0188,
    "obscene": 0.2733,
    "threat": 0.0183,
    "insult": 0.4133,
    "identity_hate": 0.0518
  },
  "Execution Time": 0.2174965309968684
}
```

#### Input:

```json
{
  "text": "You are a complete loser",
  "stemmer": "snowball"
}
```

#### Output:

```json
{
  "predictions": {
    "toxic": 0.769,
    "severe_toxic": 0.0183,
    "obscene": 0.2513,
    "threat": 0.0182,
    "insult": 0.3787,
    "identity_hate": 0.0502
  },
  "Execution Time": 0.2113154350008699
}
```

#### Example using curl command

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{ "text": "You are a complete loser", "stemmer": "lemmatize"}'

{"predictions":{"toxic":0.7986,"severe_toxic":0.0188,"obscene":0.2733,"threat":0.0183,"insult":0.4133,"identity_hate":0.0518},"Execution Time":0.13700330699793994}%
```

### License

This project is licensed under the MIT License.
