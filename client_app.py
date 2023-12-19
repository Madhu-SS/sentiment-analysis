import streamlit as st
import pandas as pd
import joblib
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import threading  # Import the threading module

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ...

# HTML template for styling
html_temp = """<h1 style ="color:gold;text-align:center;">Sentiment Analysis for Hotel Review</h1>"""
st.markdown(html_temp, unsafe_allow_html=True)

# Function to add background from URL
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.pinimg.com/736x/34/80/57/348057d60a02295353f1874b16a1b261--frankfurt-am-main-color-interior.jpg");
             background-attachment: fixed;
	     background-position: 25% 75%;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# Text area for input
st.subheader("Please write your hotel review")
text = st.text_area("0")

# Placeholder for displaying the result
result_placeholder = st.empty()

loaded_model=joblib.load('log.joblib')

vectorizer = joblib.load('vec.joblib')

def text_cleaning(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters, numbers, and extra whitespaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove punctuation
    text = ''.join([x for x in text if x not in string.punctuation])

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(['room', 'hotel','restaurant','pepole','day','night'])
    tokens = [x for x in tokens if x not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

# SQLite database connection
conn = sqlite3.connect('sentiment_data.db')
cursor = conn.cursor()

# Create a table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS sentiment_data (
        date TEXT,
        sentiment TEXT
    )
''')
conn.commit()

def btn_click():
    cleaned_input_text = text_cleaning(text)   
    X_test_vectorized = vectorizer.transform([cleaned_input_text])
    p = loaded_model.predict(X_test_vectorized)
    
    # Store sentiment in the database using a separate thread
    current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    threading.Thread(target=insert_sentiment, args=(current_date, p[0])).start()

    st.markdown(f'<div style="font-size: 34px; color:green; font-weight: bold;">Result: {p[0]}</div>', unsafe_allow_html=True)

def insert_sentiment(current_date, sentiment):
    # Function to be run in a separate thread
    with sqlite3.connect('sentiment_data.db') as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO sentiment_data (date, sentiment) VALUES (?, ?)', (current_date, sentiment))
        conn.commit()

# ...


btn = st.button("Check the Sentiment", on_click=btn_click)

# ...

# Password protection
password = st.text_input("Enter Password:", type="password")

if password == "madhu":  # Replace "your_password" with your actual password
    st.success("Password accepted! You can now access the histogram.")
    
    # Access to histogram page
    st.subheader("Histogram Page")
    
    # Date selection for histogram
    selected_date = st.date_input("Select a Date:")

    # Fetch data from the database for the selected date
    cursor.execute('SELECT sentiment FROM sentiment_data WHERE date = ?', (selected_date,))
    sentiments = cursor.fetchall()

    # Create a DataFrame for plotting
    df_sentiments = pd.DataFrame(sentiments, columns=['sentiment'])
    
    # Explicitly create a figure
    fig, ax = plt.subplots(figsize=(8, 6))

    if not df_sentiments.empty:
        # Plot histogram only if the DataFrame is not empty
        sns.countplot(x='sentiment', data=df_sentiments, ax=ax)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment') 
        plt.ylabel('Count')
    else:
        # If DataFrame is empty, still show an empty plot
        ax.set_title('No data available')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')

    # Moved outside of the conditional block
    st.pyplot(fig)

# Close the database connection
conn.close()