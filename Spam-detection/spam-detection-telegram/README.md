## Step to step implementation of the spam detector
#### We are going to create a simple telegram spam-detection bot 

<ul>
<li>Open telegram and create a telegram bot via https://t.me/BotFather</li>
<li>Copy an API_TOKEN</li>
<li>Paste an API_TOKEN instead of "API_TOKEN" below in the code</li>
<li>Run code (for example on Google Colab)</li>
</ul>

```python
!pip uninstall telebot
!pip install pyTelegramBotAPI
!pip install pytelegrambotapi --upgrade
#####
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
