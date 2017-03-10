# IJCAI17
IJCAI17 Customer Flow Forecasts on Koubei.com


- Rank of season 1 : 1088
- Rank of season 2 : 537



# TODO
- Add weather data in order to get more info
- Cluster data into different cluster as a new feature(using PCA and KMean, GMM may be slow on such a volumn data, > 60million Records)
- Change Sklearn implementation to TensorFlow version(sklearn's a little slow within more data and estimator)
- Gradient Boost Regression optimization if have time(compute take a lot time)

# Main program
1) python preprocess_data.py

2) sed -i -e 's/,,/,/g' result.txt

3) python process_result.py

4) predict.csv is the final result
