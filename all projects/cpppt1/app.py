import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("mlbots2.csv")
to_drop = ["Unnamed: 0",'peakrpm', 'compressionratio', 'stroke', 'symboling']
df.drop(df[to_drop], axis = 1, inplace = True)
X= df.drop("price",axis=1)
y= df["price"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=87)
model = LinearRegression()
model.fit(X_train.values,y_train.values)
y_pred=model.predict(X_test.values)
print(r2_score(y_pred,y_test))
print(len(X))
from flask import Flask,render_template, request
app = Flask(__name__)
app._static_folder="static"
@app.route('/',methods=['GET','POST'])
def home():
   return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        enginesize = int(request.form['enginesize'])
        curbweight = int(request.form['curbweight'])
        highwaympg = int(request.form['highwaympg'])
        horsepower = int(request.form['horsepower'])
        carwidth = int(request.form['carwidth'])
        drivewheel = int(request.form['drivewheel'])
        citympg = int(request.form['citympg'])
        cylindernumber = int(request.form['cylindernumber'])
        boreratio = int(request.form['boreratio'])
        CarName = int(request.form['CarName'])
        fueltype = int(request.form['fueltype'])
        aspiration = int(request.form['aspiration'])
        doornumber  = int(request.form['doornumber'])
        carbody = int(request.form['carbody'])
        enginelocation = int(request.form['enginelocation'])
        wheelbase = int(request.form['wheelbase'])
        carlength = int(request.form['carlength'])
        carheight = int(request.form['carheight'])
        enginetype = int(request.form['enginetype'])
        fuelsystem = int(request.form['fuelsystem'])      
        
        row = model.predict([[CarName,	fueltype,	aspiration	,doornumber	,carbody,	drivewheel,	enginelocation,	wheelbase,	carlength	,carwidth	,carheight	,curbweight,	enginetype,	cylindernumber,	enginesize	,fuelsystem	,boreratio	,horsepower,	citympg	,highwaympg]])

    return render_template('index.html',prediction="{:.2f}".format(row[0]))


if __name__ == '__main__':
    app.run(debug=True)