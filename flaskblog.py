from flask import Flask, render_template, request, url_for,flash, redirect
from forms import RegistrationForm, LoginForm
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.max_columns = None

from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

import urllib.request
import json
app = Flask(__name__)
app.config['SECRET_KEY']='a8ca03f6bb27fb5d2e9543b0a5c0ded3'
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///site.db'
db=SQLAlchemy(app)
bcrypt=Bcrypt(app)
login_manager=LoginManager(app)

@login_manager.user_loader
def load_user(user_id):
	return User.query.get(int(user_id))
class User(db.Model, UserMixin):
	id=db.Column(db.Integer,primary_key=True)
	username=db.Column(db.String(20),unique=True,nullable=False)
	email=db.Column(db.String(120),unique=True,nullable=False)
	password=db.Column(db.String(60),nullable=False)
	posts=db.relationship('Post',backref='author',lazy=True)
	def __repr__(self):
		return f"User('{self.username}','{self.email}')"

class Post(db.Model):
	id=db.Column(db.Integer,primary_key=True)
	title=db.Column(db.String(100),nullable=False)
	date_posted=db.Column(db.DateTime,nullable=False,default=datetime.utcnow)
	content=db.Column(db.Text,nullable=False)
	user_id=db.Column(db.Integer,db.ForeignKey('user.id'),nullable=False)
	def __repr__(self):
		return f"Post('{self.title}','{self.date_posted}')"

@app.route("/login",methods=['GET','POST'])
def login():
	form=LoginForm()
	if form.validate_on_submit():
		user=User.query.filter_by(email=form.email.data).first()
		if user and bcrypt.check_password_hash(user.password,form.password.data):
			login_user(user,remember=form.remember.data)
			return render_template('home.html')
		else:
			flash('Login Unsucessful')

	return render_template('login.html',title='Login',form=form)

@app.route("/register",methods=['GET','POST'])
def register():
	form=RegistrationForm()
	if form.validate_on_submit():
		hashed_password=bcrypt.generate_password_hash(form.password.data).decode('utf-8')
		user=User(username=form.username.data, email=form.email.data, password=hashed_password)
		db.session.add(user)
		db.session.commit()
		flash('Account has been created for {form.username.data}!','success')
		return redirect(url_for('login'))
	return render_template('register.html',title='Register',form=form)

@app.route("/home",methods=['GET','POST'])
def home():
	if request.method=='POST':
		mk=request.form['mk']
		url='http://www.omdbapi.com/?apikey=34035a0a&type=movie&s='
		url=url+str(mk)
		json_obj=urllib.request.urlopen(url)
		data=json.load(json_obj)
		data=data['Search']
		return render_template('about.html',data=data)
	return render_template('home.html')

    

@app.route("/about",methods=['GET','POST'])
def about():
	if request.method=='POST':
		imdb_id=request.form['imdbid']
		url='http://www.omdbapi.com/?apikey=34035a0a&i='+str(imdb_id)
		json_obj=urllib.request.urlopen(url)
		data=json.load(json_obj)
		return render_template('info.html',data=data)

@app.route("/review",methods=['GET','POST'])
def review():

	class recommend_movies(object):
	    def __init__(self):
	        self.file = pd. read_csv('data/movies_metadata.csv')
	        self.m = 0
	        self.C = 0
	        
	    def preprocessing(self):
	        file = self.file.copy()
	        file['genres'] = self.file['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x]\
	                                                                       if isinstance(x, list) else [])
	        vote_counts = file[file['vote_count'].notnull()]['vote_count'].astype('int')
	        vote_averages = file[file['vote_average'].notnull()]['vote_average'].astype('int')
	        self.C = vote_averages.mean()

	        self.m = vote_counts.quantile(0.95)

	        file['year'] = pd.to_datetime(file['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] \
	                                                                           if x != np.nan else np.nan)
	        return file
	    
	    def weighted_rating(self,x):
	        v = x['vote_count']
	        R = x['vote_average']
	        return (v/(v+self.m) * R) + (self.m/(self.m+v) * self.C)

	    def get_top_movies(self):
	        df = self.preprocessing()
	        qualified = df[(df['vote_count'] >= self.m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())]\
	                    [['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
	        qualified['vote_count'] = qualified['vote_count'].astype('int')
	        qualified['vote_average'] = qualified['vote_average'].astype('int')
	        qualified['wr'] = qualified.apply(self.weighted_rating, axis=1)
	        qualified = qualified.sort_values('wr', ascending=False).head(250)
	        qualified = qualified.sort_values('wr', ascending=False).head(250)
	        return qualified


	    def build_chart(self,genre, percentile=0.85):
	        df = self.preprocessing()
	        s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
	        s.name = 'genre'
	        gen_md = df.drop('genres', axis=1).join(s)
	        df1 = gen_md[gen_md['genre'] == genre]
	        vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
	        vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
	        C = vote_averages.mean()
	        m = vote_counts.quantile(percentile)
	        
	        qualified = df1[(df1['vote_count'] >= m) & (df1['vote_count'].notnull()) & (df1['vote_average'].notnull())]\
	                    [['title', 'year', 'vote_count', 'vote_average', 'popularity']]
	        qualified['vote_count'] = qualified['vote_count'].astype('int')
	        qualified['vote_average'] = qualified['vote_average'].astype('int')
	        
	        qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+self.m) * x['vote_average'])\
	                                          + (self.m/(self.m+x['vote_count']) * self.C), axis=1)
	        
	        return qualified
	    
	    def get_reommended_movies(self,title):
	        df = self.preprocessing()
	        links_small = pd.read_csv('data/links_small.csv')
	        links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
	        df = df.drop([19730, 29503, 35587])
	        df['id'] = df['id'].astype('int')
	        
	        smd = df.loc[df['id'].isin(links_small)]
	        smd['tagline'] = smd['tagline'].fillna('')
	        smd['description'] = smd['overview'] + smd['tagline']
	        smd['description'] = smd['description'].fillna('')
	        
	        tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
	        tfidf_matrix = tf.fit_transform(smd['description'])
	        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
	        
	        smd = smd.reset_index()
	        
	        titles = smd['title'].str.lower()
	        indices = pd.Series(smd.index, index=smd['title'].str.lower())
	        print(indices.index)
	        idx = indices[title]
	        sim_scores = list(enumerate(cosine_sim[idx]))
	        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	        sim_scores = sim_scores[1:31]
	        movie_indices = [i[0] for i in sim_scores]
	        
	        return titles.iloc[movie_indices]
	cls = recommend_movies()
	prediction = list(cls.get_reommended_movies('rustom').head(10))
	details=[]
	url='http://www.omdbapi.com/?apikey=34035a0a&type=movie&t='
	for i in prediction:
		i=i.replace(' ','%20')
		json_obj=urllib.request.urlopen(url+i)
		data=json.load(json_obj)
		details.append(data)
	return render_template('review.html',details=details)
@app.route("/hello",methods=['GET','POST'])
def hello():
	if request.method=='GET':
		render_template('hello.html')


if __name__=='__main__':
	app.run(debug=True)

