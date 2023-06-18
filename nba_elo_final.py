import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data into a pandas dataframe
df = pd.read_csv('nba_elo_games_merged-4.csv')

# Define the features used for training the model
features = np.array(['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'carm-elo1_pre', 'carm-elo2_pre', 'carm-elo_prob1', 'carm-elo_prob2', 'raptor1_pre', 'raptor2_pre', 'raptor_prob1', 'raptor_prob2'])

# Remove any missing or duplicate data
df = df.dropna(subset=features)
df.drop_duplicates(inplace=True)

# df.to_excel('cleaned_data.xlsx')

# Scale the data to a range of 0-1
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df['score1'] > df['score2'], test_size=0.2, random_state=42)

# Train the logistic regression model
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

def predict_outcome(team1, team2, date):
    # Get the ELO ratings and probabilities for the teams before the game
    new_game = df[(df['team1'] == team1) & (df['team2'] == team2) & (df['date'] < date)].sort_values(by='date', ascending=False).head(1)
    if new_game.empty:
        return None
    new_game = new_game[features]
    # Scale the data for the new game
    new_game = scaler.transform(new_game)
    # Make a prediction for the outcome of the new game
    prediction = lr.predict_proba(new_game)[0][1]
    return prediction

team1 = 'GSW'
team2 = 'DEN'
date = '2022-01-01'
prediction = predict_outcome(team1, team2, date)
if prediction is None:
    print(f"No previous data found for teams {team1} and {team2} before {date}")
else:
    print(f"The predicted outcome of the game between {team1} and {team2} on {date} is {prediction*100:.1f}% in favor of {team1} winning.")
