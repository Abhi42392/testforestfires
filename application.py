from flask import render_template,jsonify,Flask,request
from sklearn.preprocessing import StandardScaler
import pickle
application = Flask(__name__)
app=application

ridge_pickle=pickle.load(open("models/ridge.pkl","rb"))
standard_scaler=pickle.load(open("models/scaler.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict",methods=["POST","GET"])
def predict_data():
    if request.method=="POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = int(request.form.get("Classes"))
        Region = int(request.form.get("Region"))
        
        #scale new data
        new_scaled_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        #predict using Ridge
        result=ridge_pickle.predict(new_scaled_data)
        return render_template("home.html",results=result[0])
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
