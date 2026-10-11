from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/loan_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/form")
def form():
    return render_template("form.html")


@app.route("/predict", methods=["POST"])
def predict():

    Gender = int(request.form["Gender"])
    Married = int(request.form["Married"])
    Dependents = int(request.form["Dependents"])
    Education = int(request.form["Education"])
    Self_Employed = int(request.form["Self_Employed"])

    ApplicantIncome = float(
        request.form["ApplicantIncome"]
    )

    CoapplicantIncome = float(
        request.form["CoapplicantIncome"]
    )

    LoanAmount = float(
        request.form["LoanAmount"]
    )

    Loan_Amount_Term = float(
        request.form["Loan_Amount_Term"]
    )

    Credit_History = float(
        request.form["Credit_History"]
    )

    Property_Area = int(
        request.form["Property_Area"]
    )

    data = np.array([[
        Gender,
        Married,
        Dependents,
        Education,
        Self_Employed,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        Property_Area
    ]])

    prediction = model.predict(data)
    probs = model.predict_proba(data)

    print(probs)

    probability = probs[0][1]

    if prediction[0] == 1:
        result = "Loan Approved ✅"
        status = "approved"
    else:
        result = "Loan Rejected ❌"
        status = "rejected"

    
    
    return render_template(
    "result.html",
    prediction=result,
    status=status,
    probability=round(probability * 100, 2),

    accuracy=78.86,
    records=614,
    model_name="Logistic Regression"
)


if __name__ == "__main__":
    app.run(debug=True)