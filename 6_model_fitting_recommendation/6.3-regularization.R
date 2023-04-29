## Section 6.3 - Regularization

# Regularization ----------------------------------------------------------

# Motivation behind using regularization
# To improve our results, we will use regularization.

# On including the movie effects, we observed that the RMSE reduced from 1.048 to 0.986.
# Despite the large movie to movie variation, our improvement in RMSE was only about 5%. To
# understand the problem with the model that included only movie effects, let's look at the
# 10 largest mistakes predicted by the model:
library(dslabs)
test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) %>% pull(title) 

# They seem like obscure movies. Almost all of them have very extreme predicted ratings.
# To make a better sense, let's look at the top 10 worst and best movies based on b_hat_i. The
# relevant codes and the list of the titles of the top 10 best and worst movies based on b_hat_i
# are listed below:
movie_titles <- movielens %>% 
  dplyr::select(movieId, title) %>%
  distinct()

movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  dplyr::select(title, b_i) %>% 
  slice(1:10) %>%  
  pull(title)

movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  dplyr::select(title, b_i) %>% 
  slice(1:10) %>%  
  pull(title)
# The titles of the 10 best movies based on b_hat_i all seem to be obscure movies.

# It is important to look at how often these movies are rated in our data table. The code used
# for the purpose is as follows:
train_set %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:10) %>% 
  pull(n)

train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>%
  slice(1:10) %>% 
  pull(n)

# The supposed “best” and “worst” movies were rated by very few users, in most cases just 1.
# These movies were mostly obscure ones because with fewer users there's a lot of uncertainty
# in the estimation. Therefore, larger estimates of b_hat_i, negative or positive, are more
# likely.

# Large errors can increase our RMSE, so we would rather be conservative when unsure.

# Regularization permits us to penalize large estimates that are formed using small sample sizes.
# It has commonalities with the Bayesian approach that shrunk predictions.

# Penalized least squares
# Using regularization to estimate the movie effect
# To estimate the b’s, we will now minimize this equation, which contains a penalty term:
# (1/N) * the sum from u,i to n * (y_u,i - mu - b_i)^2 + λ * the sum from i to n * (b_i)^2
# The first term is the mean squared error and the second is a penalty term that gets larger
# when many b's are large.

# The values of b that minimize this equation are given by:
# b_hat_i(λ) = (1/(λ +n_i)) * the sum from u = 1 to n_i * (Y_u,i - mu_hat),
# where n_i is a number of ratings b for movie i.

# The larger λ is, the more we shrink. λ is a tuning parameter, so we can use cross-validation
# to choose it. We should be using full cross-validation on just the training set, without using
# the test set until the final assessment.

# Using λ = 3, let's see how the estimates shrink by looking at the plot of the regularized
# estimates versus the least squares estimates.

# Using the penalized estimates of b_hat_i(λ), the list of the titles of the top 10 best and the
# top 10 worst movies now make sense. The list of best movies now consists of the movies that
# are watched more and have more ratings. The list of the top 10 best and top 10 worst movies
# based on penalized estimates of b_hat_i(λ) are provided here.

# Using regularization to estimate the user effect
# We will now minimize this equation:
# (1/N) * the sum from u,i to n * (y_u,i - mu - b_i - b_u)^2 + λ *
  # (the sum from i to n * (b_i)^2 + the sum from u to n * (b_u)^2)

# The estimates that minimize the equation can be found similarly to what we did earlier for
# the regularized movie effect model equation. In this case also, to estimate λ, we should use
# full cross-validation on just the training set, without using the test set until the final
# assessment.

# Matrix Factorization ----------------------------------------------------

# Matrix factorization in the context of movie recommendation system

# Our earlier model,
# Y_u,i = mu + b_i + b_u + ∈_u,i,
# accounts for movie to movie differences through the b_i and user to user differences through
# the b_u. However, it fails to account for an important source of variation related to the
# fact that groups of movies and groups of users have similar rating patterns. We can observe
# these patterns by studying the residuals and converting our data into a matrix where each
# user gets a row and each movie gets a column:
# r_u,i = y_u,i - b_hat_i - b_hat_u,
# where y_u,i is the entry in row u and column i.

# Here for illustrative purposes, we will only consider a small subset of movies with many
# ratings and users that have rated many movies. The movie Scent of a Woman (movieId == 3252)
# is kept in the small subset because we use it for a specific example. The subset of movies
# is created using the following code:
train_small <- movielens %>% 
  group_by(movieId) %>%
  filter(n() >= 50 | movieId == 3252) %>% ungroup() %>%
  group_by(userId) %>%
  filter(n() >= 50) %>% ungroup()

y <- train_small %>% 
  dplyr::select(userId, movieId, rating) %>%
  pivot_wider(names_from = "movieId", values_from = "rating") %>%
  as.matrix()

rownames(y)<- y[,1]
y <- y[,-1]

movie_titles <- movielens %>% 
  dplyr::select(movieId, title) %>%
  distinct()

colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)])

# Our data subset can be converted to residuals by removing the column and row effects:
y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))
y <- sweep(y, 2, colMeans(y, na.rm=TRUE))

# If the model above describes all the signal and the ∈_u,i's are just noise, then the residuals
# for different movies should be independent of each other. However from the following plots
# and the table with pairwise correlation between a set of five movies, it is evident that
# there's a pattern in the data.

# Please refer to the lesson or the textbook section on matrix factorization for further study

# SVD and PCA -------------------------------------------------------------

# You can think of singular value decomposition (SVD) as an algorithm that finds the vectors p
# and q that permit us to write the matrix of residuals r with m rows and n columns in the following 
# way:
# r_u,i = p_u,1*q_1,i + p_u,2*q_2,i + ... + p_u,m*q_m,i,
# with the variability of these terms decreasing and the p's uncorrelated to each other.

# SVD also computes the variabilities so that we can know how much of the matrix’s total
# variability is explained as we add new terms.

# The vectors q are called the principal components and the vectors p are the user effects. By
# using principal components analysis (PCA), matrix factorization can capture structure in the
# data determined by user opinions about movies.

# ..Code..
y[is.na(y)] <- 0
y <- sweep(y, 1, rowMeans(y))
pca <- prcomp(y)

dim(pca$rotation)

dim(pca$x)

plot(pca$sdev)

var_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))
plot(var_explained)

library(ggrepel)
pcs <- data.frame(pca$rotation, name = colnames(y))
pcs %>%  ggplot(aes(PC1, PC2)) + geom_point() + 
  geom_text_repel(aes(PC1, PC2, label=name),
                  data = filter(pcs, 
                                PC1 < -0.1 | PC1 > 0.1 | PC2 < -0.075 | PC2 > 0.1))

pcs %>% select(name, PC1) %>% arrange(PC1) %>% slice(1:10)

pcs %>% select(name, PC1) %>% arrange(desc(PC1)) %>% slice(1:10)

pcs %>% select(name, PC2) %>% arrange(PC2) %>% slice(1:10)

pcs %>% select(name, PC2) %>% arrange(desc(PC2)) %>% slice(1:10)