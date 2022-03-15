import requests

url = "https://udacity3-prediction.herokuapp.com/inference"

data = {
    'age': 34,
    'workclass': 'Private',
    'fnlgt': 245487,
    'education': '7th-8th',
    'education-num': 4,
    'marital-status': 'Married-civ-spouse',
    'occupation': 'Transport-moving',
    'relationship': 'Husband',
    'race': 'Amer-Indian-Eskimo',
    'sex': 'Male',
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 45,
    'native-country': 'Mexico',
    'salary': '<=50K'
}

r = requests.post(url, json=data)

print(f'Status code: {r.status_code}')
assert r.status_code == 200
print(f'Response body: {r.json()}')
assert r.json() == {"prediction": "<=50K"}