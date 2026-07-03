import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

from model import gradient_descent, model, text_cleaning
# from utils import predict

def main():
    data = pd.read_csv('/Users/a0000/coding_folder/logistic regression/spam.csv')

    data['Clean_Message'] = data['Message'].apply(text_cleaning) # Cleaning Text
    data['Category'] = data['Category'].map({"spam":1, "ham":0}).astype(int)

    x =  data['Clean_Message']
    y = data['Category']

    xtrain , xtest , ytrain , ytest = train_test_split(x,y, test_size=0.2 , random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english' , max_features=5000)
    xtrain_vec = vectorizer.fit_transform(xtrain).toarray()
    xtest_vec = vectorizer.transform(xtest).toarray()


    nfeatures = xtrain_vec.shape[1]
    w_init = np.zeros(nfeatures)
    b_init = 0

    w_final, b_final, cost_history = gradient_descent(xtrain_vec, ytrain, w_init, b_init, alpha=1, numiter=10000)

    test_probs = model(xtest_vec, w_final, b_final)
    test_preds = (test_probs >= 0.5).astype(int)

    print("Actual Class:      ", ytest)
    print("Prediction Class:", test_preds)

    accuracy = np.mean(test_preds == ytest) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__=="__main__":
    main()


