
def test_welcome(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}

def test_positive_inference(client):
    item = {
        'age': 38,
        'workclass': 'Private',
        'fnlgt': 215646,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Divorced',
        'occupation': 'Handlers-cleaners',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States',
        'salary': '<=50K'
    }
    r = client.post("/inference", json=item)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_negative_inference(client):
    item = {
        'age': 52,
        'workclass': 'Self-emp-not-inc',
        'fnlgt': 209642,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 45,
        'native-country': 'United-States',
        'salary': '>50K'
    }
    r = client.post("/inference", json=item)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}