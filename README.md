# nba-elo
NBA Game Outcome Prediction:
This code uses a logistic regression model to predict the outcome of an NBA game based on the teams' ELO ratings and probabilities, as well as their CARM-Elo and Raptor ratings and probabilities. The model is trained on data from the 2015-2016 through 2019-2020 seasons and tested on data from the 2020-2021 season.


Requirements:
pandas
sklearn
numpy

Data:

The data used for this project is in the form of a CSV file, 'nba_elo_games_merged-4.csv', which contains information on the teams playing, the date of the game, the scores, and the ELO, CARM-Elo, and Raptor ratings and probabilities for the teams before the game.

Usage:

The script can be run by calling it with python3. The script will first load the data into a pandas dataframe, removing any missing or duplicate data. The data is then scaled to a range of 0-1.
Then the script will split the data into training and testing sets and train the logistic regression model.
The script will evaluate the model's accuracy using the testing set.

The function predict_outcome(team1, team2, date) can be used to predict the outcome of a game between the two teams on the specified date. The input team1, team2 are team names, date is in the format of 'yyyy-mm-dd' .

The script defines the teams and date for which the outcome is to be predicted and prints the outcome with the predicted win probability for team1.

Please make sure that the data file 'nba_elo_games_merged-4.csv' is in the same directory as the script.

Note:
If the function predict_outcome is not able to find the previous data for teams and date specified, it will raise an error "No previous data found for teams {team1} and {team2} before {date}".
