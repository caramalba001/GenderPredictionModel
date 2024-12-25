import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras

# Define Thai character set and helper function
thai_characters = ['', 'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'ะ', 'ั', 'า', 'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'เ', 'แ', 'โ', 'ใ', 'ไ', '็', '่', '้', '๊', '๋', '์']

def get_name_indices(name):
    """Converts a Thai name into a list of character indices."""
    name_indices = []
    for c in name:
        try:
            i = thai_characters.index(c)
        except ValueError:
            i = 0  # Replace unknown character with 0
        name_indices.append(i)
    return name_indices

# Load and clean data
females = open('data_male.txt', 'r', encoding='utf-8').read().split("\n")
males = open('data_female.txt', 'r', encoding='utf-8').read().split("\n")

females = list(set(females))  # Remove duplicates
males = list(set(males))

print(f'Female names: {len(females)} | Male names: {len(males)}')

# Create DataFrame
data_df = pd.concat([
    pd.DataFrame({'name': females, 'gender': 1}),  # Female: 1
    pd.DataFrame({'name': males, 'gender': 0})     # Male: 0
])

data_df['name_indices'] = data_df['name'].apply(get_name_indices)

# Pad sequences
max_len = max(data_df['name_indices'].apply(len))
X = pad_sequences(data_df['name_indices'], maxlen=max_len, padding='post')
y = data_df['gender'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = Sequential([
    Embedding(input_dim=len(thai_characters)+1, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

# Train model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks
)

# Load best model
model = keras.models.load_model("best_model.keras")

# Function to predict gender
def predict_gender(name):
    """Predicts gender of a given Thai name."""
    name_indices = pad_sequences([get_name_indices(name)], maxlen=max_len, padding='post')
    prediction = model.predict(name_indices)
    gender = 'Male' if prediction[0][0] > 0.5 else 'Female'
    probability = round(prediction[0][0]*100, 2) if gender == 'Male' else round((1-prediction[0][0])*100, 2)
    return gender, probability

# Testing predictions
test_data = open('female.txt', 'r', encoding='utf-8').read().split("\n")
predictions = [predict_gender(name) for name in test_data]

# Save results
df2 = pd.DataFrame({
    'Name': test_data,
    'Predicted Gender': [pred[0] for pred in predictions],
    'Probability': [pred[1] for pred in predictions]
})
df2.to_csv('gender_predictions.csv', index=False, encoding='utf-8')

print("Predictions saved to gender_predictions.csv")
