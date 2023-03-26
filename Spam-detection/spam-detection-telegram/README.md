## Step to step implementation of the spam detector
#### We are going to create a simple telegram spam-detection bot 

<ul>
<li>Open telegram and create a telegram bot via https://t.me/BotFather</li>
<li>Copy an API_TOKEN</li>
<li>Paste an API_TOKEN instead of "API_TOKEN" below in the code</li>
<li>Run code (for example on Google Colab)</li>
</ul>
Here i'm showing some screens and source code is in bot.py or below.

![image](https://user-images.githubusercontent.com/122405130/227812094-181ee0a0-ab82-4919-8da3-afb457cd947d.png)

![image](https://user-images.githubusercontent.com/122405130/227812957-5ba16c1b-bd4b-4583-ac31-553de5ce0b57.png)


This code loads SMS data from a remote URL using pandas, trains a Naive Bayes classifier using scikit-learn, and defines a message handler function for spam detection.

The first part of the code loads SMS data from the specified remote URL and creates a pandas DataFrame object 'data'. The DataFrame has two columns - 'label' and 'message'. 'label' column contains the labels for each message as either 'ham' or 'spam', and 'message' column contains the actual text message.

The second part of the code trains a Naive Bayes classifier using scikit-learn's MultinomialNB class. The CountVectorizer class is used to convert the text data into a numerical format that can be used for training the classifier. The 'stop_words' argument in CountVectorizer is set to 'english' to remove common English stop words from the text data. The transformed text data is then split into 'X' and 'y' variables, where 'X' contains the transformed text data and 'y' contains the labels.

The third part of the code defines a message handler function for spam detection. The function is decorated with '@bot.message_handler' which indicates that the function should be called when the bot receives a new message. The 'func=lambda message: True' argument ensures that the function is called for every message that the bot receives.

Inside the message handler function, the incoming message is preprocessed using the same CountVectorizer object used in training. The preprocessed message is then passed to the trained Naive Bayes classifier for prediction. If the prediction is 'spam', the message is considered as spam and the bot responds with a warning message 'SPAM'. If the prediction is 'ham', the message is considered as not spam, and the bot responds with an acknowledgement message "It's HAM".

Note that the commented-out line "bot.delete_message(message.chat.id, message.message_id)" is intended to delete the message if it is classified as spam, but it is currently commented out to avoid accidentally deleting non-spam messages.
```python
!pip uninstall telebot
!pip install pyTelegramBotAPI
!pip install pytelegrambotapi --upgrade

import telebot
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Set the Telegram API key
API_KEY = "YOUR_API_KEY"

# Create a Telegram bot object
bot = telebot.TeleBot(API_KEY)

# Load the SMS data from a remote URL using pandas
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
data = pd.read_csv(url, sep='\t', header=None)
data.columns = ['label', 'message']

# Train a Naive Bayes classifier using scikit-learn
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']
clf = MultinomialNB()
clf.fit(X, y)

# Define a message handler function for spam detection
# Define a message handler function for spam detection
@bot.message_handler(func=lambda message: True)
def detect_spam(message):
    # Preprocess the incoming message using the same CountVectorizer used in training
    X_message = vectorizer.transform([message.text])
    # Use the trained classifier to predict whether the message is spam or not
    prediction = clf.predict(X_message)[0]
    if prediction == 'spam':
        # If the message is classified as spam, delete it and respond with a warning
        #bot.delete_message(message.chat.id, message.message_id)
        ##bot.send_message(message.chat.id, "Warning: This message appears to be spam and has been deleted.")
         bot.send_message(message.chat.id, "SPAM")
    if prediction == 'ham':
        # If the message is classified as ham, respond with an acknowledgement
        bot.send_message(message.chat.id, "It's HAM")


# Define message handler functions for the /start, /help, /info, and /status commands
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Hello, I am a Telegram bot. Use /help to see what I can do.")

@bot.message_handler(commands=['help'])
def help(message):
    bot.reply_to(message, "I support the following commands: \n /start \n /info \n /help \n /status")

@bot.message_handler(commands=['info'])
def info(message):
    bot.reply_to(message, "I am a simple Telegram bot created using the python-telegram-bot library.")

@bot.message_handler(commands=['status'])
def status(message):
    bot.reply_to(message, "I am up and running.")

# Start the bot and continuously listen for incoming messages
print("Bot is up and running...")
bot.polling()
```
