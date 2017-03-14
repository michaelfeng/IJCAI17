# IJCAI17
IJCAI17 Customer Flow Forecasts on Koubei.com


- Rank of season 1 : 1088
- Rank of season 2 : 537


# TODO
- [Done] Add weather data in order to get more info
- [Done] Cluster data into different cluster as a new feature(using PCA and KMean, GMM may be slow on such a volumn data, > 60million Records)
- [Done] Add LightGBM algorithm for fast train and optimization
- Gradient Boost Regression optimization if have time(compute take a lot time), deffer
- Change Sklearn implementation to TensorFlow version(sklearn's a little slow within more data and estimator), deffer

# Main program
jupyter notebook run.ipynb

or

1) python preprocess_data.py

2) sed -i -e 's/,,/,/g' result.txt

3) python process_result.py

4) predict.csv is the final result


# Dataset

[Official Dataset](https://tianchi.shuju.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.YALIeW&raceId=231591)
