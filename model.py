import numpy as np
from bs4 import BeautifulSoup
import re

def text_cleaning(text):
    text = str(text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[$£€]\s?\d+", " MONEY ", text)
    text = re.sub(r"\b\d{7,}\b", " PHONE ", text)
    text = re.sub(r"\d+", " NUMBER ", text)

    text = re.sub(r"[^a-zA-Z0-9_ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def model(x,w,b):
    z = np.dot(x,w) + b

    yhat = sigmoid(z)
    return yhat

def log_loss(y, yhat):
    eps = 1e-15
    yhat = np.clip(yhat, eps, 1 - eps)

    loss = (-y * np.log(yhat)) - ((1 - y) * np.log(1 - yhat))
    cost = np.mean(loss)

    return cost
def derivatives(x,y,w,b):
    m = x.shape[0]

    z = np.dot(x,w) + b
    y_arranged = np.asarray(y).flatten()
    yhat = sigmoid(z).flatten()

    error = yhat - y_arranged

    dw = np.dot(x.T , error) / m
    db = np.sum(error) / m

    return dw , db

def gradient_descent(x,y,winit,binit,alpha,numiter):
    cost_hist = []
    w = winit
    b = binit
    for i in range(numiter):
        dw , db = derivatives(x,y,w,b)
        w -= alpha * dw
        b -= alpha * db
        if (i%100) == 0:
            yhat = model(x, w, b)
            cost = log_loss(y, yhat)
            cost_hist.append(cost)
    return w , b, cost_hist

