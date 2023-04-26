## Section 3.2 - Smoothing

# Smoothing ---------------------------------------------------------------

#Smoothing is a very powerful technique used all across data analysis. It is designed to detect
#trends in the presence of noisy data in cases in which the shape of the trend is unknown. 

#The concepts behind smoothing techniques are extremely useful in machine learning because
#conditional expectations/probabilities can be thought of as trends of unknown shapes that we
#need to estimate in the presence of uncertainty.

# ..Code..
# see that the trend is wobbly
library(tidyverse)
set.seed(1)
n <- 100
x <- seq(-pi*4, pi*4, len = n)
tmp <- data.frame(x = x , f = sin(x) + x/8, e = rnorm(n, 0, 0.5)) 
p1 <- qplot(x, f, main = "smooth trend", ylim = range(tmp$f+tmp$e), data = tmp, geom = "line")
p2 <- qplot(x, e, main = "noise", ylim = range(tmp$f+tmp$e), data = tmp, geom = "line")
p3 <- qplot(x, f+e, main = "data = smooth trend + noise", ylim = range(tmp$f+tmp$e), data = tmp, geom = "line")
gridExtra::grid.arrange(p1, p2, p3)

# estimate the time trend in the 2008 US popular vote poll margin
library(tidyverse)
library(dslabs)
data("polls_2008")
qplot(day, margin, data = polls_2008)

# use regression to estimate
resid <- ifelse(lm(margin~day, data = polls_2008)$resid > 0, "+", "-")
polls_2008 %>% 
  mutate(resid = resid) %>% 
  ggplot(aes(day, margin)) + 
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  geom_point(aes(color = resid), size = 3)

# Bin Smoothing -----------------------------------------------------------

# The general idea of smoothing is to group data points into strata in which the value of f(x)
# can be assumed to be constant. We can make this assumption because we think f(x) changes slowly
# and, as a result, f(x) is almost constant in small windows of time. 

# In mathematical terms, the assumption implies:
# E[Y_i | X_i = x_i] ≈ mu if |x_i - x_0| <=
 
# This assumption implies that a good estimate for f(x) is the average of the Y_i values in the
# window. The estimate is:
# f_hat(x_0) = (1)/(N_0) and the sum from i∈A_0 of Y_i
 
# In smoothing, we call the size of the interval |x - x_0| satisfying the particular condition
# the window size, bandwidth, or span.
 
# The bin smoother approach can be thought of as a weighted averge - mathematically, it is this:
# f_hat(x_0) = the sum from 1 to N of w_0(x_i)Y_i
  
# ..Code..
# bin smoothers
span <- 3.5
tmp <- polls_2008 %>%
  crossing(center = polls_2008$day) %>%
  mutate(dist = abs(day - center)) %>%
  filter(dist <= span) 

tmp %>% filter(center %in% c(-125, -55)) %>%
  ggplot(aes(day, margin)) +   
  geom_point(data = polls_2008, size = 3, alpha = 0.5, color = "grey") +
  geom_point(size = 2) +    
  geom_smooth(aes(group = center), 
              method = "lm", formula=y~1, se = FALSE) +
  facet_wrap(~center)

# larger span
span <- 7 
fit <- with(polls_2008, 
            ksmooth(day, margin, kernel = "box", bandwidth = span))

polls_2008 %>% mutate(smooth = fit$y) %>%
  ggplot(aes(day, margin)) +
  geom_point(size = 3, alpha = .5, color = "grey") + 
  geom_line(aes(day, smooth), color="red")

# kernel
span <- 7
fit <- with(polls_2008, 
            ksmooth(day, margin, kernel = "normal", bandwidth = span))

polls_2008 %>% mutate(smooth = fit$y) %>%
  ggplot(aes(day, margin)) +
  geom_point(size = 3, alpha = .5, color = "grey") + 
  geom_line(aes(day, smooth), color="red")

# Local Weighted Regression (loess) ---------------------------------------

# A limitation of the bin smoothing approach is that we need small windows for the approximately
# constant assumptions to hold which may lead to imprecise estimates of f(x). Local weighted
# regression (loess) permits us to consider larger window sizes.

# One important difference between loess and bin smoother is that we assume the smooth function
# is locally linear in a window instead of constant.

# The result of loess is a smoother fit than bin smoothing because we use larger sample sizes
# to estimate our local parameters.

# ..Code..
# The full code can be found in the .R file Handout at the beginning of the section.
polls_2008 %>% ggplot(aes(day, margin)) +
  geom_point() +
  geom_smooth(color="red", span = 0.15, method = "loess", method.args = list(degree=1))

# Beware of Default Smoothing Parameters ----------------------------------

# Local weighted regression (loess) permits us to fit parabola by considering a larger
# window size than the one considered while fitting a line.

# ..Code..
# The full code can be found in the .R file Handout at the beginning of the section.

total_days <- diff(range(polls_2008$day))
span <- 28/total_days
fit_1 <- loess(margin ~ day, degree=1, span = span, data=polls_2008)
fit_2 <- loess(margin ~ day, span = span, data=polls_2008)


polls_2008 %>% mutate(smooth_1 = fit_1$fitted, smooth_2 = fit_2$fitted) %>%
  ggplot(aes(day, margin)) +
  geom_point(size = 3, alpha = .5, color = "grey") +
  geom_line(aes(day, smooth_1), color="red", lty = 2) +
  geom_line(aes(day, smooth_2), color="orange", lty = 1)


polls_2008 %>% ggplot(aes(day, margin)) +
  geom_point() +
  geom_smooth()

# Connecting Smoothing to Machine Learning --------------------------------

# In the 2 vs 7 example, we saw that linear regression was not flexible enough to capture
# the non-linear nature of p(x_1,x_2). Smoothing approaches may provide an improvement in
# capturing the same.