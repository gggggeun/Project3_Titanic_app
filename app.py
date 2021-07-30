# importing the packages
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, session, logging
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
# from sklearn.externals import joblib
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Loading model so that it works on production
model = joblib.load('./model/model.pkl')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Task %r>' % self.id


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/post', methods=['POST','GET'])
def post():
    if request.method == 'POST':
        task_content = request.form['content']
        new_task = Todo(content=task_content)

        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect('/post')
        except:
            return 'There was an issue adding your task'

    else:
        tasks = Todo.query.order_by(Todo.date_created).all()
        return render_template('post.html', tasks=tasks)


@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = Todo.query.get_or_404(id)

    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/post')
    except:
        return 'There was a problem deleting that task'

@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    task = Todo.query.get_or_404(id)

    if request.method == 'POST':
        task.content = request.form['content']

        try:
            db.session.commit()
            return redirect('/post')
        except:
            return 'There was an issue updating your task'

    else:
        return render_template('update.html', task=task)





class PredictorsForm(Form):
	"""
	This is a form class to retrieve the input from user through form

	Inherits: request.form class
	"""
	p_class = StringField(u'머무르실 선실 등급의 숫자를 입력해주세요.(등급 : 1,2,3)', validators=[validators.input_required()])
	sex = StringField(u'성별이 여자면 0, 남자면 1을 입력해주세요.', validators=[validators.input_required()])
	age = StringField(u'나이를 입력해주세요.(ex. 24)', validators=[validators.input_required()])
	sibsp = StringField(u'함께 승선했을 것 같은 형제 또는 배우자의 수를 입력해주세요. (ex. 2)', validators=[validators.input_required()])
	parch = StringField(u'함께 탑승했을 것 같은 부모 또는 자녀의 수를 입력해주세요. (입력값 : 0~6)', validators=[validators.input_required()])
	fare = StringField(u'지불할 탑승요금을 입력해주세요 (입력값 : 0~100)', validators=[validators.input_required()])
	embarked = StringField(u'타이타닉호를 탑승할 장소의 숫자를 입력해주세요. (0:Cherbourg, 1:Southampton, 2:Queenstown)', validators=[validators.input_required()])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	form = PredictorsForm(request.form)
	
	# Checking if user submitted the form and the values are valid
	if request.method == 'POST' and form.validate():
		# Now save all values passed by user into variables
		p_class = form.p_class.data
		sex = form.sex.data
		age = form.age.data
		sibsp = form.sibsp.data
		parch = form.parch.data
		fare = form.fare.data
		embarked = form.embarked.data

		# Creating input for model for predictions
		predict_request = [int(p_class), int(sex), float(age), int(sibsp), int(parch), float(fare), int(embarked)]
		predict_request = np.array(predict_request).reshape(1, -1)

		# Class predictions from the model
		prediction = model.predict(predict_request)
		prediction = str(prediction[0])

		# Survival Probability from the model
		predict_prob = model.predict_proba(predict_request)
		predict_prob = str(predict_prob[0][1])

		# Passing the predictions to new view(template)
		return render_template('predictions.html', prediction=prediction, predict_prob=predict_prob)

	return render_template('predict.html', form=form)

@app.route('/train', methods=['GET'])
def train():
	# reading data
	df = pd.read_csv("./data/titanic.csv")

	#defining predictors and label columns to be used
	predictors = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
	label = 'Survived'

	#Splitting data into training and testing
	df_train, df_test, y_train, y_test = train_test_split(df[predictors], df[label], test_size=0.20, random_state=42)

	# Data cleaning and filling missing values
	age_fillna = df_train.Age.mean()
	embarked_fillna = df_train.Embarked.value_counts().index[0]
	cabin_fillna = 'NA'

	# filling missing values in training data
	df_train.Age = df_train.Age.fillna(df.Age.mean())
	df_train.Embarked = df_train.Embarked.fillna(embarked_fillna)
	df_train.Cabin = df_train.Cabin.fillna(cabin_fillna)

	# filling missing values imputed from training set to avoid data leakage
	df_test.Age = df_test.Age.fillna(df.Age.mean())
	df_test.Embarked = df_test.Embarked.fillna(embarked_fillna)
	df_test.Cabin = df_test.Cabin.fillna(cabin_fillna)

	# Label encoding of object type predictors
	le = dict()
	for column in df_train.columns:
	    if df_train[column].dtype == np.object:
	        le[column] = LabelEncoder()
	        df_train[column] = le[column].fit_transform(df_train[column])

	# Applying same encoding from training data to testing data
	for column in df_test.columns:
	    if df_test[column].dtype == np.object:
	        df_test[column] = le[column].transform(df_test[column])

	# Initializing the model
	model = RandomForestClassifier(n_estimators=25, random_state=42)

	# Fitting the model with training data
	model.fit(X=df_train, y=y_train)

	# Saving the trained model on disk
	joblib.dump(model, './model/model.pkl')

	# Return success message for user display on browser
	return 'Success'




if __name__ == "__main__":
    app.run(debug=True)



