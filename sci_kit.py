import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


#step 1: load the data
df = pd.read_csv('Data.csv')
# print(df)
# print('columns')
# print(df.columns)

#step 2: Features and labels
X = df['message']
Y = df['category']

# print(X)
# print(Y)

# Step 3: Convert text to numeric using CountVectorizer
'''
fit() builds the vocabulary
It scans all the messages and assigns an index to each unique word
| Word   | Index |
| ------ | ----- |
| need   | 0     |
| help   | 1     |
| login  | 2     |
| server | 3     |
| urgent | 4     |
| ...    | ...   |

This vocabulary is stored inside the vectorizer:
vectorizer.vocabulary_

2. transform() encodes each message into a sparse matrix
Every message is converted into a row of numbers, where each column is the count of a word.

So the matrix X_vectorized shape is:
[number_of_messages, number_of_unique_words]

Most messages don’t contain all words in the vocabulary. So we don’t store the entire big matrix — we only store (row, column) = count where the count is non-zero.
That's called a sparse matrix, and it saves memory.
Example full matrix:
'''
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
print(X_vectorized)

# vectorizer = CountVectorizer()
# Y_vectorized = vectorizer.fit_transform(Y)
# print(Y_vectorized)

'''
🔸 WHY ONLY X_vectorized AND NOT Y?
➤ X_vectorized (input features):
These are your messages like "Need help with login".

But ML models like MultinomialNB need numerical input, so we use CountVectorizer() to convert those text messages into a matrix of word counts.

That’s what you pass to .fit() and .predict().

➤ Y (labels / targets):
These are already in label/text form, like "Technical Issue", "Urgent Request".
The model doesn’t need to vectorize Y because it learns which combinations of words in X map to each category in Y.

So:

X_vectorized = numerical features (input to the model)

Y = categories (targets the model should learn to predict)
'''

# Step 4: Split into train and test
X_train, X_test, Y_train,Y_test = train_test_split(X_vectorized, Y, test_size=0.3, random_state=42)

# print(X_train)

# print("**********")
# print(X_test)

# print("&&&&&&&&&&&&&")
# print(Y_train)

# print("!!!!!!!!!!!!!!")
# print(Y_test)


# Step 5: Train the model
model = MultinomialNB()
model.fit(X_train, Y_train)

# Step 6: Predict on test data
'''
🔸 WHEN YOU PREDICT
Y_pred = model.predict(X_test)
You’re passing the vectorized test messages (X_test) to the trained model, and it predicts the most likely category (Y_pred) for each.
'''
Y_pred = model.predict(X_test)
#print(Y_pred)

# Step 7: Accuracy
print("Accuracy: ", accuracy_score(Y_test,Y_pred))


# Step 8: Test new messages
test_messages = ["System Failure!Please help Immediately", "How do i reset my password", "What time does your office open?"]
test_vectors = vectorizer.transform(test_messages)
predictions = model.predict(test_vectors)


for msg,cat in zip(test_messages,predictions):
    print(f"Message: '{msg}' => Category: '{cat}' ")
