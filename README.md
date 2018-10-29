# PUBG-Win-Prediction
In a PUBG game, up to 100 players start in each match (matchId). Players can be on teams (groupId) which get ranked at the end of the game (winPlacePerc) based on how many other teams are still alive when they are eliminated. In game, players can pick up different munitions, revive downed-but-not-out (knocked) teammates, drive vehicles, swim, run, shoot, and experience all of the consequences -- such as falling too far or running themselves over and eliminating themselves.

We create a model which predicts players' finishing placement based on their final stats, on a scale from 1 (first place) to 0 (last place).

## Installation

### Requirements
* Python 3.3+
* Numpy
* Pandas
* Sklearn
* LightGBM
* [PUBG Dataset] (https://www.kaggle.com/c/pubg-finish-placement-prediction/data)

`$ pip3 install numpy`
`$ pip3 install pandas`
`$ pip3 install sklearn`
`$ pip3 install lightgbm`

