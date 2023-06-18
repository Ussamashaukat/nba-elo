import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings


warnings.filterwarnings("ignore")


# load data
df = pd.read_csv("nba_elo_games_merged-4.csv")

# select relevant columns
columns_to_keep = ['elo1_pre', 'elo2_pre', 'carm-elo1_pre', 'carm-elo2_pre', 'raptor1_pre', 'raptor2_pre', 'score1', 'score2', 'ortg', 'drtg', 'total_opp', 'home_opp', 'won']
df = df[columns_to_keep]

# df = df.dropna(subset=['carm-elo1_pre', 'carm-elo2_pre', 'raptor1_pre', 'raptor2_pre', 'score1', 'score2', 'total_rating'],how='any')
df = df.dropna(subset=columns_to_keep)

scaler = StandardScaler()

# split data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.4, random_state=42)

# define X and y for training and test sets
X_train = train_df.drop(columns=['won'])
y_train = train_df['won']
X_test = test_df.drop(columns=['won'])
y_test = test_df['won']

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# make predictions on test set
y_pred = lr.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)



# Now you can use this model to predict the outcome of future games
# by providing the appropriate input to the predict() function

future_games = pd.read_csv("future_games.csv")
future_games_X = future_games[columns_to_keep]

future_games_X= future_games_X.drop(columns=['won'])

future_games_predictions = lr.predict(future_games_X)

print(future_games_predictions)