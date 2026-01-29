# Intro-To-Statistics-With-Python
Statistics with Python

#### What is Statistics? ####
# the field of statistics - the practice and study of collecting and analyzing data
# a summary statistic - a fact about or summary of some data

# What can statistics do?
# Can answer a bunch of questions:
# How likely is someone to purchase a product?
# how many occupants will your hotel have?
# how many sizes of jeans need to be manufactured so they can fit 95% of the population?
# A/B tests: which ad is more effective in getting people to purchase a product?

# What statistics can't do:
# Why is Game of Thrones so popular? (people might lie or leave out reasons)
# correlation vs. causation

# Types of statistics
# Descriptive stats
# describes and summarizes data
# ex. % breakdown of how people commute to work

# Inferential stats
# use a sample of data to make inferences about a larger population

# 2 Types of data:
# 1. Numeric (quantitative)
# 1a. Continuous (measured) such as airplane speed and time waiting in line
# 1b. Discrete (counted) such as # of pets or # of packages shipped
# 2. Categorical (qualitative)
# 2a. Nominal (unordered) - no inherent order (marriage status, country of residence)
# 2b. Ordinal (ordered) - has an inherent order, like a survey question where you have to select 'strongly disagree','somewhat agree', 'neither agree nor disagree', etc
# sometimes categorical data can be represented as numbers

# Why does data type matter?
# it matters because it helps us to know which sum stats and visualizations to use

### Measure of Center ####
# Histograms
# Takes a bunch of data points and separates them into bins, or ranges of values
# three different definitions, or measures, or center: mean, median, and mode

# Mean, or average, common way to summarizing data
# in Python, we can use numpy's mean function, passing it the variable of interest
import numpy as np
np.mean(msleep['sleep_total']

# Median - the value where 50% of the data is lower than it, and 50% of the data is higher than it
# we sort first, and then take the middle one, which is index 41 in this example
msleep['sleep_total'].sort_values()
msleep['sleep_total'].sort_values().iloc[41]
# in Python, we can use np.median to do the calculation for us:
np.median(msleep['sleep_total'])

# Mode - most frequent value in data
msleep['sleep_total'].value_counts()
msleep['vore'].value_counts()
# another option
import statistics
statistics.mode(msleep['vore'])
# mode is often used for categorical variables

# Adding an outlier
# subset msleep to select rows where 'vore' equals 'insecti'
msleep[msleep['vore'] == 'insecti']
msleep[msleep['vore'] == "insecti"]['sleep_total'].agg([np.mean, np.median])
# the mean goes down by more than 3 hours in this example, while the median changed by less than an hour

# Which measure to use?
import matplolib.pyplot as plt
data['value'].hist()
plt.show()
# since the mean is more sensitive to extreme values, it works better for symmetrical data
# however, if the data is skewed, meaning it's not symmatrical, like this, median is usually better to use
# left-skewed and right-skewed data
# the mean and median are different when data is skewed
# the mean is pulled in the direction of the skew
# the mean is lower than the median on the left-skewed data, and higher than the median on the right-skewed data
# because the mean is pulled around by extreme values, it is best to use the median since it is less affected by outliers

# Import numpy with alias np
import numpy as np

# Subset country for USA: usa_consumption
usa_consumption = food_consumption[food_consumption['country'] == "USA"]

# Calculate mean consumption in USA
print(np.mean(usa_consumption['consumption']))

# Calculate median consumption in USA
print(np.median(usa_consumption['consumption']))

# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Subset for food_category equals rice
rice_consumption = food_consumption[food_consumption['food_category'] == 'rice']

# Histogram of co2_emission for rice and show plot
rice_consumption['co2_emission'].hist()
plt.show()

# Subset for food_category equals rice
rice_consumption = food_consumption[food_consumption['food_category'] == 'rice']

# Calculate mean and median of co2_emission with .agg()
print(rice_consumption['co2_emission'].agg([np.mean, np.median]))

#### Measures of spread ####
# Describes how spread apart of close the data are
# Variance - measures the avg distance from each data point to the data's mean

# Calculating Variance
# 1. Subtract mean from each data point
dists = msleep['sleep_total'] - np.mean(msleep['sleep_total'])
print(dists)

# 2. Square each distance
sq_dist = dists ** 2
print(sq_dists)

# 3. Sum square distances
sum_sq_dists = np.sum(sp_dists)
print(sum_sq_dists)

# 4. Divide by number of data points - 1
variance = sum_sq_dists/(83-1)
print(variance)

# the higher the variance, the more spread out the data is
# we can calculate the variance in one step using np.var, setting the ddof argument to 1.

# Using np.var()
np.var(msleep['sleep_total'], ddof = 1)

# without ddof = 1, population variance is calculated instead of sample variance:
np.var(msleep['sleep_total'])

# Standard Deviation - calculated by getting the square root of the variance
np.sqrt(np.var(msleep['sleep_total'], ddof = 1))
np.std(msleep['sleep_total'], ddof = 1)

# Mean absolute deviation
# takes the absolute value of the distances to the mean, and then takes the mean of those differences
# similar to STD, but not the same
dists = msleep['sleep_total'] - np.mean(msleep['sleep_total'])

# Standard Deviation vs. Mean Absolute Deviation
# STD squares distances, penalizing longer distances more than shorter ones
# MAD penalizes distance equally
# one isn't better than the other, but SD is more common than MAD

# Quantiles - also called percentiles, split up the data into some # of equal parts
np.quantile(msleep['sleep_total'], 0.5)
# this example ^ gives us 10.1 hours, and means that 50% of mammals in the data set sleep <10.1 hours
# the other 50% sleep more than 10.1 hours
# this is the same as the median

# we can also pass in a list of #s to get the multiple quantiles at once
np.quantile(msleep['sleep_total'], [0, 0.25, 0.5, 0.75, 1])
# this example ^ splits data into four equal parts, also called quartiles

# Boxplots use quartiles
import matplotlib.pyplot as plt
plt.boxplot(msleep['sleep_total'])
plt.show()

# Splitting the data in five equal pieces
np.quantile(msleep['sleep_total'], [0, 0.2, 0.4, 0.6, 0.8, 1])
# Another option - takes starting #, stopping #, and the # intervals
np.linspace(start, stop, num)

# We can compute the same quantiles using np.linspace starting at 0, stopping at 1, and splitting into 5 intervals
np.quantiles(msleep['sleep_total'], np.linspace(0, 1, 5))

# Interquartile Range (IQR)
# Distance between the 25th and 75th percentile, which is also the height of the box in a boxplot
np.quantile(msleep['sleep_total'], 0.75) - np.quantile(msleep['sleep_total'], 0.25)
from scipy.stats import iqr
iqr(msleep['sleep_total'])

# Outliers - data points that are substantially different from others
# How do we know what 'substantially different' means?
# Rule of thumb:
data < Q1 - 1.5 x IQR
# or
data > Q3 + 1.5 x IQR

# Finding outliers
# we start by calculating the IQR of the mammals' body weights in this example
from scipy.stats import iqr
iqr = iqr(msleep['bodywt'])
lower_threshold = np.quantile(msleep['bodywt'], 0.25) - 1.5 * iqr
upper_threshold = np.quantile(msleep['bodywt'], 0.75) + 1.5 * iqr

# we can now subset the DataFrame to find mammals whose body weight is below or above thresholds
msleep[(msleep['bodywt'] < lower_threshold) | (msleep['bodywt'] > upper_threshold)]

# All in one go
msleep['bodywt'].describe()

# Print variance and sd of co2_emission for each food_category
print(food_consumption.groupby('food_category')['co2_emission'].agg(['var','std']))

# Create histogram of co2_emission for food_category 'beef'
food_consumption[food_consumption ['food_category'] == 'beef']['co2_emission'].hist()
plt.show()

# Create histogram of co2_emission for food_category 'eggs'
plt.figure()
food_consumption[food_consumption['food_category'] == 'eggs']['co2_emission'].hist()
plt.show()

# Calculate the quartiles of co2_emission
print(np.quantile(food_consumption['co2_emission'], [0, 0.25, 0.5, 0.75, 1]))

# Calculate the quintiles of co2_emission
print(np.quantile(food_consumption['co2_emission'], [0, 0.2, 0.4, 0.6, 0.8, 1]))

# Calculate the deciles of co2_emission
print(np.quantile(food_consumption['co2_emission'], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))

# Calculate total co2_emission per country: emissions_by_country
emissions_by_country = food_consumption.groupby('country')['co2_emission'].sum()

# Calculate total co2_emission per country: emissions_by_country
emissions_by_country = food_consumption.groupby('country')['co2_emission'].sum()

# Compute the first and third quantiles and IQR of emissions_by_country
q1 = np.quantile(emissions_by_country, 0.25)
q3 = np.quantile(emissions_by_country, 0.75)
iqr = q3 - q1

# Calculate the lower and upper cutoffs for outliers
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

# Subset emissions_by_country to find outliers
outliers = emissions_by_country[(emissions_by_country < lower) | (emissions_by_country > upper)]

#### What are the chances? ####
# What are the chances of closing a sale?
# What are the chances of rain tomorrow?

# Measuring Chance
# What is the probability of an event?
# P(event) = # of ways event can happen/total # of possible outcomes
# Coin flip example: P(heads) = 1 way to get heads/2 possible outcomes = 1/2 = 50%
# Probability is always between 0 and 100% 
# 0 means impossible, 100 means guaranteed

# Sampling from a DataFrame
# by default, it randomly samples one row from the DataFrame
# if we run a second time, we might get another row
sales_counts.sample()

# Setting a random seed
# this ensures we get the same results when we run the script
np.random.seed(10)
# Python's random # generator starting point
# and orienting number, generates same value each time
# number doesn't matter
# the only thing that matters is that we use the same seed so that we get the same result

# Sampling without replacement (meetings are happening on same day)
# we aren't replacing the name we already pulled out (in this "who is going to the meeting?" example)
# the first time we picked someone, the chances were one in four
# now, since we are not replacing the person already picked, each person has a 1 in 3 chance of being picked
# to recreate this in Python, we can pass 2 into the sample method, which will give us 2 rows of the DataFrame
sales_counts.sample(2)

# Sampling with replacement (because the meetings are happening on different days)
# Everyone goes back to 1 in 4 chances of being picked
sales_counts.sample(5, replace = True)

# Independent events
# two events are independent if the probability of the 2nd event isn't affected by the outcome of the first event
# sampling with replacement - each pick is independent

# Dependent events
# two events are dependent if the probability of the second event is affected by the outcome of the first event
# sampling without replacement - each pick is dependent

# Count the deals for each product
counts = amir_deals['product'].value_counts()

# Calculate probability of picking a deal with each product
# .shape[0] gives the count of rows, which is equal to the total number of deals Amir worked on
probs = counts/amir_deals.shape[0]
print(probs)

# Set random seed
np.random.seed(24)

# Sample 5 deals without replacement
sample_without_replacement = amir_deals.sample(5)
print(sample_without_replacement)

# Set random seed
np.random.seed(24)

# Sample 5 deals with replacement
sample_with_replacement = amir_deals.sample(5, replace = True)
print(sample_with_replacement)

#### Discrete Distributions ####
# Rolling the dice - 6 possible outcomes, 1/6 or 17% chance of being rolled
# describes the probability of each possible outcome in a scenario
# expected value: mean of a probability distribution - calculated by multiplying each value by its probability and summing
# the expected value of rolling a fair die is 3.5
# Barplot visualization

# Probability = area
P(die roll) =< 2 = ?
# we can calculate probabilities of different outcomes by taking areas of the probability distribution
# what is the probability that our die roll is less than or equal to 2?

# Uneven die that contains 2 3s
# we have a 0% chance of getting a 2 and a 33% chance of getting a 3
# to calculate the expected value of this die, we now multiply 2 by 0 since it is impossible to get a 2 and 3 by its new probability, 1/3
# gives us a value that is slightly higher than the fair die

# Adding areas 
P(uneven die roll) =< 2 = ?
# 1/6 probability of getting 1 and 0 probability of getting 2
# this sums to 1/6

# the probability distributions we have seen so far are both discrete probability distribution (they have discrete outcomes)
# when all outcomes have same probability, it is called discrete uniform distribution

# Sampling from discrete distributions
print(die) # shows probability of each outcome on a fair die
np.mean(die['number'])

# We will sample from it 10 times to simulate 10 rolls
# We are sampling with replacement so that we're sampling from the same distribution each time
rolls_10 = die.sample(10, replace = True)
rolls_10

# Visualizing a sample
# we can visualize the outcomes of the ten rolls using a histogram, defining the bins we want using np.linspace
rolls_10['number'].hist(bins=np.linspace(1,7,7))
plt.show()

# Sample distribution vs. theoretical distribution
# sample of 10 rolls
np.mean(rolls_10['number]) = 3.0

# theoretical probability distribution
mean(die['number']) = 3.5

# the mean of our sample isn't super close to what we were expecting (3.0 vs 3.5)
# but if we roll the die 100 times, the distribution of the rolls looks a bit more even, and the mean is close to 3.5
# rolling a 1000 times makes it look even more like the theoretical
# this is called the Law of Large Numbers

# Creating a probability distribution
# Remember that expected value can be calculated by multiplying each possible outcome with its corresponding probability and taking the sum
# linspace expects at most three main arguments: the start, the stop, and the # of samples

# Create a histogram of restaurant_groups and show plot
restaurant_groups['group_size'].hist(bins = [2, 3, 4, 5, 6])
plt.show()

# Create probability distribution
size_dist = restaurant_groups['group_size'].value_counts() / restaurant_groups.shape[0]
# Reset index and rename columns
size_dist = size_dist.reset_index()
size_dist.columns = ['group_size', 'prob']

# Expected value
# expected_value is scalar (a single number), not a DataFrame or Series
expected_value = np.sum(size_dist['group_size'] * size_dist['prob'])

# Subset groups of size 4 or more
groups_4_or_more = size_dist[size_dist['group_size'] >= 4]

# Sum the probabilities of groups_4_or_more
prob_4_or_more = groups_4_or_more['prob'].sum()
print(prob_4_or_more)

#### Continuous Distributions ####
# How can we model continuous variables?
# Waiting for the bus example
# We can model with a probability distribution
# We can use a continuous line called the continuous uniform distribution
# What is the probability that we will wait between 4 and 7 minutes?
# How about 7 minutes or less?
from scipy.stats import uniform
uniform.cdf(7,0,12) # pass at 7, set 0 and 12 for lower and upper limits; we get 58% in this example

# Greater Than probabilities
# we need to take 1 minus the probability of waiting less than 7 minutes
from scipy.stats import uniform
1 - uniform.cdf(7,0,12) # .41

# Calculating the probability of waiting between 4 and 7 minutes
from scipy.stats import uniform
uniform.cdf(7,0,12) - uniform.cdf(4,0,12) # gives us 25%

# Total area = 1
# To calculate the probability of waiting between 0 and 12 minutes, we multiply 12 by 1/12, which is 1, or 100%
# this makes sense since we are certain we will wait between 1 and 12 minutes

# Generating random numbers according to uniform distribution
# .rvs takes in the min value, max value, followed by the # of random values we want to generate
from scipy.stats import uniform
uniform.rvs(0,5, size = 10) # generates 10 random values between 0 and 5

# Min and max wait times for back-up that happens every 30 min
min_time = 0
max_time = 30

# Import uniform from scipy.stats
from scipy.stats import uniform

# Calculate probability of waiting less than 5 mins
prob_less_than_5 = uniform.cdf(5,0,30)
print(prob_less_than_5)

# Calculate probability of waiting more than 5 mins
prob_greater_than_5 = 1 - uniform.cdf(5,0,30)
print(prob_greater_than_5)

# Calculate probability of waiting 10-20 mins
prob_between_10_and_20 = uniform.cdf(20,0,30) - uniform.cdf(10,0,30)
print(prob_between_10_and_20)

# Set random seed to 334
np.random.seed(334)

# Import uniform
from scipy.stats import uniform

# Generate 1000 wait times between 0 and 30 mins
wait_times = uniform.rvs(0, 30, size=1000)

# Create a histogram of simulated times and show plot
plt.hist(wait_times)
plt.show()

#### The binomial distribution ####
# Coin flipping
# Binary outcomes - only two possible outcomes
# 1/0, win/loss, True/False

# A single flip
# the argument is called size, or # of trials
binom.rvs(# of coins, probability of heads/success, size=# of trials)
from scipy.stats import binom
binom.rvs(1, 0.5, size = 1)

# one flip many times
# flips 1 coin with 50% chance of successful 8 times
binom.rvs(1, 0.5, size=8) 

# many flips one time
# swaps the first and last argument; flips 8 coins 1 time) # give us 1 #, which is total # of heads or successes
binom.rvs(8, 0.5, size=1) 

# many flips many times
# we can pass 3 as the first argument, and set size = 10 to flip 3 coins
# returns 10 numbers, each representing the total number of heads from each of flips

# Other probabilities
# for example, having a coin that is heavier on one side than the other, so the probability of getting heads is only 25%
binom.rvs(3, 0.5, size=10)

# Binomial distribution
# probability distribution of the # of successes in a sequence of independent trials
# it can tell us the probability of getting some # of heads in a sequence of coin flips
# this is a discrete distribution
# n: total # of trials
# p: probability of success
binom.rvs(n=10, p=0.5, size=20)
# we have the biggest chance of getting 5 heads total, and a much smaller chance of getting 0 heads or 10 heads

# What is the probability of 7 heads?
P(heads=7)
# binom.pmf(num heads, num trials, prob of heads)
binom.pmf(7, 10, 0.5)

# What is the probability of 7 or fewer heads?
# binom.cdf gives the probability of getting a # of successes less than or equal to the first argument
p(heads =< 7)
binom.cdf(7, 10, 0.5)

# What is the probability of more than 7 heads?
P(heads > 7)
1-binom.cdf(7, 10, 0.5)

# Expected Value
# n * p
# Expected # of heads out of 10 flips = 10 x 0.5 = 5

# Independence
# the binomial distribution is a probability distribution of the # of successes in a sequence of independent trials
# the outcome of one trials shouldn't have an effect on the next
# if trials are not independent, the binomial distribution doesn't apply
# not binomial distribution if the probabilities of a second trial are altered due to outcome of the first

# Import binom from scipy.stats
from scipy.stats import binom

# Set random seed to 10
np.random.seed(10)

# Simulate a single deal
print(binom.rvs(1, .3, size=1))

# Simulate 1 week of 3 deals
print(binom.rvs(3, .3, size=1))

# Simulate 52 weeks of 3 deals
deals = binom.rvs(3, .3, size=52)

# Print mean deals won per week
print(np.mean(deals))

# Probability of closing 3 out of 3 deals
prob_3 = binom.pmf(3, 3, .3)

# Probability of closing <= 1 deal out of 3 deals
prob_less_than_or_equal_1 = binom.cdf(1, 3, .3)

# Probability of closing > 1 deal out of 3 deals
prob_greater_than_1 = 1-binom.cdf(1,3,.3)

# Expected number won with 30% win rate
won_30pct = 3 * .3
print(won_30pct)

# Expected number won with 25% win rate
won_25pct = 3 * .25
print(won_25pct)

# Expected number won with 35% win rate
won_35pct = 3 * .35
print(won_35pct)

#### The Normal Distribution ####
# bell curve
# symmatrical
# the area beneath the curve is 1
# curve never hits 0
# described by mean and standard deviation

# Standard Normal Distribution -> when the mean is 0 and the standard deviation is 1
# 68% of the area is within 1 standard deviation of the mean
# 95% of the area falls within 2 standard deviations of the mean
# 99.7% of the area fall within 3 stds
# sometimes called 58-95-99-97 rule

# Example
# What percent of women are shorter than 154cm?
from scipy.stats import norm
norm.cdf(154, 161, 7)
# we pass in the number of interest, 154, followed by the mean and std of the distribution we are using
# this give us 16% of woman are shorter than 154cm

# What percent of women are taller than 154cm?
from scipy.stats import norm
1 - norm.cdf(154, 161, 7)
# this gives us 84% of women are taller than 154cm

# What percent of women are 154-157cm?
norm.cdf(157, 161, 7) - norm.cdf(154, 161, 7)
# this gives us 12.5%

# What height are 90% of women shorter than?
# we pass 0.9 into norm.ppf along with same mean and std that we've been working with
norm.ppf(0.9, 161, 7)
# this tells us that 90% of women are shorter than 170cm tall

# What height are 90% of women taller than? This is also the height that 10% of women are shorter than
norm.ppf((1-0.9), 161, 7)

# Generating random numbers
# just like with other distributions, we can generate random numbers from a normal distribution using norm.rvs
# we pass the distribution's mean and std, and the sample size we want
# generate 10 random heights
norm.rvs(161, 7, size = 10)

# Histogram of amount with 10 bins and show plot
amir_deals['amount'].hist(bins=10)
plt.show()

# Probability of deal < 7500
prob_less_7500 = norm.cdf(7500, 5000, 2000)
print(prob_less_7500)

# Probability of deal > 1000
prob_over_1000 = 1 - norm.cdf(1000, 5000, 2000)
print(prob_over_1000)

# Probability of deal between 3000 and 7000
prob_3000_to_7000 = norm.cdf(7000, 5000, 2000) - norm.cdf(3000, 5000, 2000)
print(prob_3000_to_7000)

# Calculate amount that 25% of deals will be less than
pct_25 = norm.ppf(.25, 5000, 2000)
print(pct_25)

# Calculate new average amount
new_mean = 5000 + (5000*.20)

# Calculate new standard deviation
new_sd = 2000 + (2000*.3)

# Simulate 36 new sales
new_sales = norm.rvs(new_mean, new_sd, size = 36)

# Create histogram and show
plt.hist(new_sales)
plt.show()

#### The Central Limit Theorem ####
# Dice rolling example
die = pd.Series([1, 2, 3, 4, 5, 6])
# Simulating rolling the die 5 times
samp_5 = die.sample(5, replace = True)
print(samp_5) # prints 'array([3,1,4,1,1])
# Now take the mean
np.mean(samp_5) # prints 2.0

# Rolling the dice 5 times 10 times
# Repeat 10 times:
# Roll 5 itimes
# take the mean
# Use a for loop
# we loop from 0 to 9 so that the process is repeated 10 times
# inside the loop, we roll 5 times and append the sample's mean to the sample_means list
sample_means = []
for i in range(10):
    samp_5 = die.sample(5, replace = True)
    sample_means.append(np.mean(samp_5))
# this gives us a list of 10 different sample means
# [3.8, 4.0, 3.8, 3.6, 3.2, 4.8, 2.6, 3.0, 2.6, 2.0]

# Sampling distributions
# 100 sample means
sample_means = []
for i in range (100):
    sample_means.append(np.mean(die.sample(5, replace = True)))
# resembles normal distribution

# 1000 sample means
sample_means = []
for i in range (1000):
    sample_means.append(np.mean(die.sample(5, replace = True)))
# resembles normal distribution even more
# THIS PHENOMENOM IS KNOWN AS THE CENTRAL LIMIT THEOREM
# CLT -> THE SAMPLING DISTRIBUTION OF A STATISTIC BECOMES CLOSER TO THE NORMAL DISTRIBUTION AS THE # OF TRIALS INCREASES
# CLT only works if the samples are random and independent

# STD & CLT
sample_sds = []
for i in range(1000):
    sample_sds.append(np.std(die.sample(5, replace = True)))
    
# Proportions & the CLT 2.44 mark
