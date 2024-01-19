# Time-series classification

## Dataset structure
We are given a multivariate dataset that consists of 7 ordered features. We have 68528 data points for each feature. Here is an overview of the 7 ordered features of the data frame:

<img width="654" alt="image" src="https://github.com/irfanmasoudi/anndl2/assets/6355974/93fb9bd9-b446-4fff-a9ee-c5c0442b661a">

We have tried 2 scenarios: 

a) Divide the dataset into train and test.
 
b) Feed all examples into the training process and do validation. When we tried to implement the second scenario into a simple model, we got much better results. So for the next experiment, we constantly used all datasets for the training process in the model selection.

We first had to be careful that the dataset was clean and preprocessed. That means no wrong or NaN values or significant time gaps in the data. ANNs look at the numerical value of the input so after we normalized all training inputs to be within the same range.

## Model Selection

Being a multivariate time series forecasting problem, we have to provide a prediction for each time step in the test prediction window. The networks we have implemented mostly used the Mean Average Error (MAE) as a metrics. The metric used to evaluate models and place the Teams in Leaderboard on Codalab is the Root Mean Squared Error (RMSE).

### Unchanged value of Window, Stride, and Telescope.
In this section, we have implemented a couple of different model scenarios. We always keep the window equal to 200, stride equal 20, and telescope with 50.

### Result in CodaLab

| Model (with window=200, stride=20 and telescope=50)            | Result (RMSE)  |
| -------------------------------------------------------------- | -------------- |
| Bidirectional LSTM with two-layer LSTM different units         | 176.7613220215 |
| One-Dimension CNN                                              | 27.7518043518  |
| Simple GRU                                                     | 17.8791122437  |
| Simple LSTM                                                    | 16.9558753967  |
| Bidirectional LSTM without cascade                             | 16.2943630219  |
| Bidirectional LSTM with single-layer LSTM and single-layer GRU | 13.6941833496  |
| Bidirectional LSTM with two-layer LSTM                         | 11.5225048065  |

### Changed value of Window, Stride, and Telescope

| Bidirectional LSTM with two-layer LSTM                 | Result (RMSE) |
| ------------------------------------------------------ | ------------- |
| 1.  Model with window=200, stride=10 and telescope=100 | 10.0559186935 |
| 2.  Model with window=200, stride=10 and telescope=100 | 9.1587381363  |
| 3.  Model with window=600, stride=2 and telescope=864  | 4.7141537666  |
| 4.  Model with window=600, stride=1 and telescope=864  | 4.0347905159  |
