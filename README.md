# PCA-Deep-Dive

The idea behind Principal component anaylsis (PCA) is to take a set of data in $Z$-dimensional space, and capture as much of the information (variance) in these data as possible using fewer dimensions. This allows us to greatly reduce the problem space, in particular for high dimensions where most of the variation can be explained through a few linear combinations of features. This process is analogous to lossy compression, where if we use all the available dimensions, decompressing will fully recover the original data, whereas with a limited number of linear combinations, we are able to capture almost the entire information with fewer data points, but unable to fully recover the original data after decompression. After we have transformed (compress) our original inputs using the learned PCA model, we can reconstruct ('unzip') it to get a simplified version of said original data, but this reconstruction loses information that pertained to components that we did not use, so it is not 100% faithful to the original. The difference between the original data compared to the 'zipped then unzipped' data is what we refer to as 'reconstruction error' and is often calculated using the Mean Squared Error (MSE) as follows:

$$MSE=\frac{\sum_{i=0}^{N}(X_{i}-\hat{X}_{i})^2}{N}$$




