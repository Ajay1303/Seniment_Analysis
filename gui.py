import tkinter as tk
from tkinter import messagebox
import joblib  # Assuming you used joblib to save your model
import re
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

# Load your pre-trained model
model = pickle.load(open('/home/ajay/Desktop/mini_project_gui/trained_model.sav', 'rb'))

# Load your CountVectorizer or any other transformer if you used one
vectorizer = pickle.load(open('/home/ajay/Desktop/mini_project_gui/vectorizer.sav', 'rb'))

import nltk
nltk.download('stopwords')

port_stem = PorterStemmer()

def preprocess_tweet(tweet):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', tweet)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content 

# Function to predict sentiment
def predict_sentiment():
    tweet = entry.get("1.0", "end-1c")  # Get text from the Text widget
    if tweet.strip() == "":
        messagebox.showwarning("Input Error", "Please enter a tweet.")
        return
    
    cleaned_tweet = preprocess_tweet(tweet)
    tweet_vector = vectorizer.transform([cleaned_tweet])  # Vectorize the tweet
    
    prediction = model.predict(tweet_vector)[0]
    
    if prediction == 1:
        result.set("☺ Positive Tweet")
        result_label.config(fg="green")
    else:
        result.set("☹️ Negative Tweet")
        result_label.config(fg="red")

# Function to clear input and result
def clear_text():
    entry.delete('1.0', tk.END)
    result.set("")



# Set up the GUI environment
root = tk.Tk()
root.title("Twitter Sentiment Analysis")
root.geometry('500x400')  # Set window size
root.configure(bg='#C4D7FF')  # Set background color (mixture of #FFF7D1 and #FFECC8)

# Add a title label
title_label = tk.Label(root, text="Twitter Sentiment Analysis", font=("Helvetica", 18, "bold"), bg='#C4D7FF', fg='#333')
title_label.pack(pady=20)

# Add a label for the tweet input
input_label = tk.Label(root, text="Enter a Tweet:", font=("Helvetica", 12), bg='#C4D7FF')
input_label.pack()

# Textbox to enter tweet
entry = tk.Text(root, width=50, height=5, font=("Helvetica", 12), bd=2)
entry.pack(pady=10)

# Button to predict sentiment
predict_button = tk.Button(root, text="Predict Sentiment", font=("Helvetica", 12), command=predict_sentiment, bg='#007BFF', fg='white', bd=0, padx=20, pady=5)
predict_button.pack(pady=15)

# Button to clear input and output
clear_button = tk.Button(root, text="Clear", font=("Helvetica", 12), command=clear_text, bg='#6c757d', fg='white', bd=0, padx=20, pady=5)
clear_button.pack(pady=5)

# Label to display the prediction result
result = tk.StringVar()
result_label = tk.Label(root, textvariable=result, font=("Helvetica", 16, "bold"), bg='#C4D7FF')
result_label.pack(pady=20)

# Start the GUI loop
root.mainloop()
