

Modeling Crypto-Currency Trends with Recurrent Neural Networks using Long Short Term Memory
-------------------------------------------------------------------------------------------

Description
---------------
 - This project is an effort to see trends in the price of crypto-currencies. While Crypto is volitile and
   almost any attempt to predict the future of crypto is essentially a random walk, I believe modeling shorter
   trends could produce accurate results. These results used in conjunction with a trading method or bot could
   aide strategy. By no means, however, is this project something that should be used as the sole source of 
   analysis for trading of any kind. Please use at your own risk.
   

Data 
-----
 - The data is normalized to reflect the percent change in price in a sequence of closing prices. The data is
   taken from https://www.coindesk.com/price/  and can be found in the /Data directory. The data was split 90/10
   for train/test respectively.
   

Hardware
----------
- Ubuntu 16.04 LTS
- Intel i5-6600 3.9 GHz quad-core with 6MB cache
- 32 Gb DDR4 at 2133 MHz

Software
----------
- Python3, Keras, Tensorflow, Numpy, Matplotlib


Models
-------
- The model I used is the Sequential Model in the Keras library with a Tensorflow backend. I connected the model
  with layers from the Long Short Term Memory package of the recurrent neural network library from Keras. This
  type of network is commonly used for time series evaluations. Included between each layer is a dropout layer
  as this helps prevent over fitting of the data. Two distinct models, one shallow and one deep,  were used in 
  the final testing each with varying layers.
  
Results
--------
- To test the accuracy of the trend prediction, I used an evaluation method that took the error for each prediction
  of the model and averaged the error over the course of a trend. With this evaluation method I was able to compare
  the error rates of multiple models at multiple different trend lengths. For the specific results of the project
  please see the results.csv along with the accompanying graphs for each of the models. The results comma separated
  values file will connect each of the different tests with a specific graph and their respective results.


