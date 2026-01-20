from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
# بارگذاری مدل ذخیره شده
model = joblib.load('iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # ورودی باید به صورت [5.1, 3.5, 1.4, 0.2] باشد
    ## We added reshaping to ensure the data is in the right format
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    # We changed the host to 0.0.0.0 and port to 80
    app.run(host='0.0.0.0', port=80)