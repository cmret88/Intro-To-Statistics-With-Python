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

# Sampling without replacement
# we aren't replacing the name we already pulled out (in this "who is going to the meeting?" example)
# the first time we picked someone, the chances were one in four
# now, since we are not replacing the person already picked, each person has a 1 in 3 chance of being picked
