# monkeytype

In depth analysis and modeling of personal typing test data from
[monkeytype](<https://monkeytype.com/>)
with an interactive streamlit
[jbreffle.github.io/monkeytype-app](<https://jbreffle.github.io/monkeytype-app>).

## Usage notes for local development

- Set up your python environment based on ./requirements.txt
- Replace the data in ./data/raw/ with your own data downloaded from your monkeytype [account](https://monkeytype.com/account) using the "Export CSV" button
- Run the notebook as desired
- Note: monkeytype only retains the 1000 most recent trials, so to store all of your results you must periodically download your data
- Note: The "Export CSV" button on the monkeytype.com/account page will export only the data you are currently viewing, so make sure "all" is selected under the "filters" option in order to export all of your data

## Notebooks

- 1_explore: What does the data look like when we apply basic visualizations and analysis?
- 2a_z_scoring: Can we compare trials with different difficulties by z-scoring by trial-type?
- 2b_learning_curve: Can we fit a learning curve to the data?
- 3a_sim_simple: What is the intrinsic correlation between accuracy and performance if we model typing as consisting of random errors with random error-correction times?
- 3b_sim_poisson: Does modeling typing as a Poisson process produce the same results?
- 4_nn_predict: Can we train a neural network to predict WPM performance from trial-type labels and number of trials completed?
- 5_nn_hyperopt: Using hyperopt to optimize the hyperparameters of the neural network
- state-space: Can we fit a state-space model to the data that takes into account trial-type difficulty to predict performance across time?
- stationary-performance: What would we expect the data to look like if there were no long-run improvement in performance?
- z_dynamodb_example.ipynb: Testing out reading/writing to a DynamoDB table
- z_s3_example.ipynb: Testing out reading/writing to an S3 bucket

## TODOs

- Calculate wpm v acc correlation for each trial type
- Run parameter grids in the simulation analyses
- Try simpler models of skill analysis: XGBregressor, fit a combined Learning Curve model
- Finalize some sort of "skill" metric that accounts for wpm difference by trial type
- Finish state_space analysis
- Finish stationary_performance analysis
- Correlate performance with health data from watch: sleep score, resting heart rate, stress levels, air quality, etc.
- Try out TimeGPT <https://arxiv.org/pdf/2310.03589.pdf> or other transformer for skill analysis
