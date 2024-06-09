from model.model import model


user_input1 ={
    'age': 55,
    'sex': 1,
    'cp': 0,
    'trestbps': 140,
    'chol': 240,
    'fbs': 0,
    'restecg': 0,
    'thalach': 170, 
    'exang': 0,
    'oldpeak': 2.1,
    'slope': 1,
    'ca': 0,
    'thal': 3,
   }



def predict(user_input):
    model_input = [list(user_input.values())]
    model_output = int(model.predict(model_input)[0])
    return model_output
