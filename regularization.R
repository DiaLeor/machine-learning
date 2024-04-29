options(digits=7)

# An education expert is advocating for smaller schools. The expert bases
# this recommendation on the fact that among the best performing schools,
# many are small schools. Let's simulate a dataset for 1000 schools. First,
# let's simulate the number of students in each school, using the following code:
set.seed(1986)
n <- round(2^rnorm(1000, 8, 1))

# Now let's assign a true quality for each school that is completely independent
# from size. This is the parameter we want to estimate in our analysis. The
# true quality can be assigned using the following code:
set.seed(1)
mu <- round(80 + 2*rt(1000, 5))
range(mu)
schools <- data.frame(id = paste("PS",1:1000),
                      size = n,
                      quality = mu,
                      rank = rank(-mu))

# We can see the top 10 schools using this code: 
library(dplyr)
schools %>% top_n(10, quality) %>% arrange(desc(quality))

# Now let's have the students in the school take a test. There is random
# variability in test taking, so we will simulate the test scores as normally
# distributed with the average determined by the school quality with a
# standard deviation of 30 percentage points. This code will simulate the
# test scores:

set.seed(1)
mu <- round(80 + 2*rt(1000, 5))

scores <- sapply(1:nrow(schools), function(i){
  scores <- rnorm(schools$size[i], schools$quality[i], 30)
  scores
})
schools <- schools %>% mutate(score = sapply(scores, mean))


# Q1 ----------------------------------------------------------------------

# What are the top schools based on the average score? Show just the ID, size,
# and the average score.

# Examine the data 
str(schools)
head(schools)

# Select the ID, size, and score
schools <- schools %>% select(id,size,score)
head(schools)

# Report the ID of the top school and average score of the 10th school.
scores_desc <- schools %>% arrange(desc(score)) # arrange by descending scores 

# What is the ID of the top school?
scores_desc$id[1] 

# What is the average score of the 10th school (after sorting from highest to
# lowest average score)?
scores_desc$score[10] 

#..Answer Code..
schools %>% top_n(10, score) %>% arrange(desc(score)) %>% select(id, size, score)

# Q2 ----------------------------------------------------------------------

# Compare the median school size to the median school size of the top 10 schools
# based on the score.

# What is the median school size overall?
median(schools$size)

# What is the median school size of the of the top 10 schools based on the score?
schools %>% top_n(10, score) %>% arrange(desc(score))  %>%
  summarize(median = median(size))

# Q3 ----------------------------------------------------------------------

# From this analysis, we see that the worst schools are also small. Plot the
# average score versus school size to see what's going on. Highlight the top
# 10 schools based on the true quality.
library(ggplot2)
schools %>% ggplot(aes(size, score)) +
  geom_point(alpha = 0.5) +
  geom_point(data = filter(schools, rank<=10), col = 2)

# What do you observe?

# The standard error of the score has larger variability when the school is
# smaller, which is why both the best and the worst schools are more likely
# to be small.

# Q5 ----------------------------------------------------------------------

# Let's use regularization to pick the best schools. Remember regularization
# shrinks deviations from the average towards 0. To apply regularization here,
# we first need to define the overall average for all schools, using the
# following code:
overall <- mean(sapply(scores, mean))
overall

# Then, we need to define, for each school, how it deviates from that average.

# Write code that estimates the score above the average for each school but
# dividing by n + a instead of n, with n the school size and alpha a
# regularization parameter. Try alpha = 25.

# What is the ID of the top school with regularization?
 
# What is the regularized score of the 10th school (after sorting from highest
# to lowest regularized score)? 

#..Answer Code..
alpha <- 25
score_reg <- sapply(scores, function(x)  overall + sum(x-overall)/(length(x)+alpha))
schools %>% mutate(score_reg = score_reg) %>%
  top_n(10, score_reg) %>% arrange(desc(score_reg))

# Q6 ----------------------------------------------------------------------
# Notice that this improves things a bit. The number of small schools that are
# not highly ranked is now lower. Is there a better alpha? Using values of alpha
# from 10 to 250, find the alpha that minimizes the RMSE.

#..Answer Code..
alphas <- seq(10,250)
rmse <- sapply(alphas, function(alpha){
  score_reg <- sapply(scores, function(x) overall+sum(x-overall)/(length(x)+alpha))
  sqrt(mean((score_reg - schools$quality)^2))
})
plot(alphas, rmse)
alphas[which.min(rmse)]

# Q7 ----------------------------------------------------------------------

# Rank the schools based on the average obtained with the best  from Q6. Note
# that no small school is incorrectly included.
alpha <- 135
score_reg <- sapply(scores, function(x)  overall + sum(x-overall)/(length(x)+alpha))
schools %>% mutate(score_reg = score_reg) %>%
  top_n(10, score_reg) %>% arrange(desc(score_reg))

# What is the ID of the top school now?

# What is the regularized average score of the 10th school now?

# Q8 ----------------------------------------------------------------------

# A common mistake made when using regularization is shrinking values towards
# 0 that are not centered around 0. For example, if we don't subtract the
# overall average before shrinking, we actually obtain a very similar result.
# Confirm this by re-running the code from the exercise in Q6 but without
# removing the overall mean.

# What value of alpha gives the minimum RMSE here?
alphas <- seq(10,250)
rmse <- sapply(alphas, function(alpha){
  score_reg <- sapply(scores, function(x) sum(x)/(length(x)+alpha))
  sqrt(mean((score_reg - schools$quality)^2))
})
plot(alphas, rmse)
alphas[which.min(rmse)]
