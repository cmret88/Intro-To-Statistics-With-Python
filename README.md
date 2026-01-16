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
