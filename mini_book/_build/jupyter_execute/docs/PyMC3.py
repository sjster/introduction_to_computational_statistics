## Introduction to PyMC3

### Attribution

It is important to acknowledge the authors who have put together fantastic resources that have allowed me to make this notebook possible. 


1. A lot of the examples here are taken from the book 'Introduction to statistical modeling and probabilistic programming using PyMC3 and ArviZ', Second Edition by Osvaldo Martin

2. [PyMC3 website](docs.pymc.io)

3. Bayesian Methods for Hackers by Davidson-Pilon Cameron

4. Doing Bayesian Data Analysis by John Kruschke

### What is PyMC3?

PyMC3 is a probabilistic programming framework for performing Bayesian modeling and visualization. It uses Theano as a backend. It has algorithms to perform Monte Carlo simulation as well as Variational Inference. 

It can be used to infer values of parameters of model equations that we are unsure about by utilizing the observed data. A good example is given here [https://docs.pymc.io/notebooks/ODE_API_introduction.html](https://docs.pymc.io/notebooks/ODE_API_introduction.html). We are trying to estimate the parameters of air resistance from the Ordinary Differential Equation (ODE) of freefall. We have an understanding of the physics behind freefall as represented by the ODE and we have observed/measured some variables but we don't know what the parameter of air resistance is here. We can use PyMC3 to perform inference and give us a distribution of potential values of air resistance. A key point to note here is that the more information we have regarding other variables, the more certainty we have in our desired variable (air resistance). Suppose we are unsure about the gravitational constant used in the ODE (implemented by specifying a prior distribution as opposed to a constant value of 9.8), we get more uncertainty in the air resistance variable as well.

We will look at an example of Linear Regression to illustrate the fundamental features of PyMC3.

### General Structure of PyMC3

It consists of phenomena represented by equations made up of Random Variables and Deterministic variables. The random variables can be divided into Observed variables and Unobserved variables. The observed variables are those for which we have data and the unobserved variables are those for which we have specify a prior distribution.

#### Observed Variables

```
with pm.Model():
    obs = pm.Normal('x', mu=0, sd=1, observed=np.random.randn(100))
```

#### Unobserved Variables

```
with pm.Model():
    x = pm.Normal('x', mu=0, sd=1)
```


### An example with Linear Regression

import pymc3
pymc3.__version__

%matplotlib inline
import arviz as az
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import graphviz
import os

os.environ['OMP_NUM_THREADS'] = '4'

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.linspace(0, 1, size)
X2 = np.linspace(0,.2, size)

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

import pymc3 as pm
from pymc3.backends import SQLite
from pymc3 import Model, Normal, HalfNormal
from pymc3 import find_MAP

basic_model = Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = Normal('alpha', mu=0, sd=5)
    beta = Normal('beta', mu=0, sd=5, shape=2)
    sigma = HalfNormal('sigma', sd=4)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2 
    
    # Deterministic variable, to have PyMC3 store mu as a value in the trace use
    # mu = pm.Deterministic('mu', alpha + beta[0]*X1 + beta[1]*X2)
    
    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

pm.model_to_graphviz(basic_model)

**Plate notation**

A way to graphically represent variables and their interactions in a probabilistic framework.

[Wikipedia Plate Notation](https://en.wikipedia.org/wiki/Plate_notation)

#### MAP Estimate 

PyMC3 computes the MAP estimate using numerical optimization, by default using the BFGS algorithm. These provide a point estimate which may not be accurate if the mode does not appropriately represent the distribution.

map_estimate = find_MAP(model=basic_model, maxeval=10000)
map_estimate

#### Distribution Information through Traceplots

from pymc3 import NUTS, sample
from scipy import optimize

with basic_model:
    
    # obtain starting values via MAP
    start = find_MAP(fmin=optimize.fmin_powell)

    # instantiate sampler - not really a good practice
    step = NUTS(scaling=start)

    # draw 2000 posterior samples
    trace = sample(2000, step, start=start)

You can also pass a parameter to step that indicates the type of sampling algorithm to use such as

* Metropolis
* Slice sampling
* NUTS

PyMC3 can automatically determine the most appropriate algorithm to use here, so it is best to use the default option.

trace['alpha']

from pymc3 import traceplot

traceplot(trace)

We will look at the summary of the sampling process. The columns will be explained as we progress through this course.

az.summary(trace)

from pymc3 import summary
summary(trace)

### Composition of Distributions for Uncertainty

You can do the same without observations, to perform computations and get uncertainty quantification. In this example add two normally distributed variables to get another normally distributed variable. By definition

$$\mu_c = \mu_a + \mu_b $$

$$\sigma_c^2 = \sigma_a^2 + \sigma_b^2$$

basic_model2 = Model()

with basic_model2:

    # Priors for unknown model parameters
    alpha = Normal('alpha', mu=20, sd=5)
    beta = Normal('beta', mu=10, sd=5, shape=1)
    # These aren't used in the calculation, but please experiment by composing various combinations
    # of these function for calculating mu
    sigma = HalfNormal('sigma', sd=1)
    σ_obs = pm.HalfCauchy("σ_obs", beta=1, testval=0.1)

    # Expected value of outcome
    mu = pm.Deterministic('mu', alpha + beta) 
    
    trace = sample(15000)
    
print(trace['mu'])
traceplot(trace)    

# Note the mean and standard deviation of the variable mu. We didn't need observations to compute this uncertainty.
try:
    # older syntax
    pm.plot_posterior(trace, var_names=['alpha', 'beta', 'mu'], credible_interval=0.68)
except:
    pm.plot_posterior(trace, var_names=['alpha', 'beta', 'mu'], hdi_prob=0.68)

### GRADED EVALUATION (5 mins)

1. What type of algorithms does PyMC3 support?

   a. MCMC
   
   b. Variational Inference
   
   c. Both (C)
   
   
2. We can mix Deterministic and Probabilistic variables in PyMC3

   a. True (C)
   
   b. False

#### HDI, HPD and ROPE

HDI, HPD and ROPE are essentially used for making decisions from the posterior distribution.
We will plot the posterior of the beta distribution with the set parameters and a credible interval for the Highest-Posterior Density (HPD) which is the interval that has the given probability indicated by the HPD. What is the probability of getting a value given by x? We can't really calculate this exactly but we can compute this probability within a range given by x + $\Delta$x, x - $\Delta$x. A related term is the Highest Density Interval (HDI) which is a more general term that can apply for any distribution such as a prior. In other words a posterior's HDI is called the HPD. 

As an example, if we suspect that the dice used at a casino is loaded, we can infer the probability of getting the value 3 from the six possible outcomes. Ideally, this should be 1/6 = 0.16666. If this happens to fall in the HPD, we can assume that the dice is fair however it may be that the distribution may be biased to one side or the other.  

Sometimes, instead of looking at the probability that x = 0.16666, we look at the probability that it falls within the range 0.12 and 0.20 called the Region of Practical Equivalence or ROPE. This implies that, based on our subjective opinion, getting a value between 0.12 and 0.20 is practically equivalent to getting a 0.16666 or that we can assume that the dice is fair given any value within this range. ROPE allows us to make inferences about an event. After computing the posterior our ROPE given by 0.12 and 0.20 can either overlap with the HPD from the posterior density of getting a 3 

* completely
* not overlap at all 
* partially overlap with the HPD

Complete overlap suggests that our computed probability coincides with what we would expect from a fair dice. If it does not overlap, it is not a fair dice and a partial overlap indicates that we cannot be certain that is either fair or unfair.

In short, we define a ROPE based on our subject matter expertise and compare it to the HPD to make a decision from the posterior distribution.

##### Credible intervals vs. Confidence Intervals

This deserves special mention particularly due to the subtle differences stemming from the Bayesian (credible intervals) vs. Frequentist (confidence intervals) approaches involved. Bayesian's consider the parameters to be a distribution and that there is no true parameter however Frequentists assume that there exists a true parameter. 
* Confidence intervals quantify our confidence that the true parameter exists in this interval. It is a statement about the interval. 
* Credible intervals quantify our uncertainty about the parameters since there are no true parameters in a Bayesian setting. It is a statement about the probability of the parameter.

For e.g. if we are trying to estimate the R0 for COVID-19, one could say that we have a 95% confidence interval of the true R0 being between 2.0 and 3.2. In a Bayesian setting, the 95% credible interval of (2,3.2) implies that we are 95% certain that 

We will see how we can visualize the following using ArViz

* HDI (Black lines)
* ROPE (Green lines)

import numpy as np
from scipy import stats as stats 

np.random.seed(1)

try:
    az.plot_posterior({'θ':stats.beta.rvs(5, 5, size=20000)},
                  credible_interval=0.75,                    # defaults to 94%
                  #hdi_prob = 0.85,  
                  rope =[0.45, 0.55])
except:                      
    az.plot_posterior({'θ':stats.beta.rvs(5, 5, size=20000)},
                      #credible_interval=0.75,                    # defaults to 94%
                      hdi_prob = 0.85,                            # credible_interval is deprecated, use hdi_prob
                      rope =[0.45, 0.55])

Another way to do this is by plotting a reference value on the posterior. Below, a reference value of 0.48 is used and it can be seen that 45.2% of the posterior is below this value while 54.8% of the posterior is above this value. If we were estimating our parameter to have a value of 0.48, this suggests a good fit but with a slight right bias or in other words that the parameter is likely to have a value greater than 0.48.

try:
    az.plot_posterior({'θ':stats.beta.rvs(5, 5, size=20000)},
                  credible_interval=0.75,                    # defaults to 94%
                  #hdi_prob = 0.85,                            # credible_interval is deprecated, use hdi_prob
                  ref_val=0.48)
except:
    az.plot_posterior({'θ':stats.beta.rvs(5, 5, size=20000)},
                      #credible_interval=0.75,                    # defaults to 94%
                      hdi_prob = 0.85,                            # credible_interval is deprecated, use hdi_prob
                      ref_val=0.48)

### Graded Evaluation (15 min)

1. HDI and HPD are the same

   a. True
   
   b. False (C)
   
2. HPD is used for making decisions from the posterior distribution

   a. True (C)
   
   b. False 
   
3. ROPE is a subjective but informed interval to help make decisions from the posterior distribution

   a. True
   
   b. False
   
4. In order to confirm our hypothesis that we have the right estimate for our parameter, we want our ROPE and the HPD to have

   a. complete overlap (C)
   
   b. partial overlap
   
   c. no overlap
   
5. A reference value can be used to indicate tge direction of bias in our posterior dsitribution

   a. True (C)
   
   b. False
   
<br>
<br>
<hr style="border:2px solid blue"> </hr>

### Modeling with a Gaussian Distribution

Gaussians (Normal distributions) are normally used to approximate a lot of practical data distributions. One reason for this is the Central Limit Theorem, which states: 

**The distribution of the sample means will be a normal distribution**

[Intuitive explanation of the Central Limit Theorem](https://statisticsbyjim.com/basics/central-limit-theorem/)

which implies that if we take the mean of the sample means, we should get the true population mean. There is also a more subtle reason for adopting the Gaussian distribution to represent a lot of phenomena; a lot of them are a result of averages of varying factors. 

The other reason is the mathematical tractability of the distribution, it is easy to compute in closed form. While not every distribution can be approximated with a single Gaussian distribution, we can use a mixture of Gaussians to represent other multi-modal distributions.

The probability density for a Normal distribution in a single dimension is given by:

$P(x) = \dfrac{1}{\sigma \sqrt{2 \pi}} e^{-(x - \mu)^2 / 2 \sigma^2}$

where $\mu$ is the mean and $\sigma$ is the standard deviation. In higher dimensions, we have a vector of means and a covariance matrix. 

#### Example with PyMC3

We read the chemical shifts data, and plot the density to get an idea of 
the data distribution. It looks somewhat like a Gaussian so maybe we can start
there. We have two parameters to infer, that is the mean and the standard deviation. 
We can estimate a prior for the mean by looking at the density and putting 
some bounds using a uniform prior. The standard deviation is however chosen to 
have a mean-centered half-normal prior - half-normal since the standard deviation 
cannot be negative. We can provide a hyperparameter for this by inspecting the
density again. These values decide how well we converge to a solution so good 
values are essential for good results. 

data = np.loadtxt('data/chemical_shifts.csv')
az.plot_kde(data, rug=True)
plt.yticks([0], alpha=0)

data

import pymc3 as pm
from pymc3.backends import SQLite, Text
model_g = Model()

with model_g:
    
    #backend = SQLite('test.sqlite')
    db = pm.backends.Text('test')
    μ = pm.Uniform('μ', lower=40, upper=70)
    σ = pm.HalfNormal('σ', sd=10)
    y = pm.Normal('y', mu=μ, sd=σ, observed=data)
    trace_g = pm.sample(draws=1000) # backend = SQLite('test.sqlite') - Does not work

az.plot_trace(trace_g)
pm.model_to_graphviz(model_g)    

<br>
<br>
<hr style="border:2px green solid "> </hr>

##### Note on scalablility 
If your trace information is too big as a result of too many variables or the model being large, you do not want to store this in memory since it can overrun the machine memory. Persisting this in a DB will allow you to reload it and inspect it at a later time as well. For each run, it appends the samples to the DB (or file if not deleted).

help(pm.backends)

from pymc3.backends.sqlite import load

with model_g:
    #trace = pm.backends.text.load('./mcmc')
    trace = pm.backends.sqlite.load('./mcmc.sqlite')
    
print(len(trace['μ']))

<br>
<br>
<hr style="border:2px green solid "> </hr>

#### Pairplot for Correlations

Use a pairplot of the parameters to ensure that there are no correlations that would adversely affect the sampling process.

az.plot_pair(trace_g, kind='kde', fill_last=False)

az.summary(trace_g)

#### Posterior Predictive Check

We can draw samples from the inferred posterior distribution to check to see how they line up with the observed values. Below, we draw 100 samples of length corresponding to that of the data from this posterior. You are returned a dictionary for each of the observed variables in the model. You can also plot the distribution of these samples by passing this variable 'y_pred_g' as shown below. Setting `mean=True` in the call to `plot_ppc` computes the mean distribution of the 100 sampled distributions and plots it as well.

Two things can be noted here:

* The mean distribution of the samples from the posterior predictive distribution is close to the distribution of the observed data but the mean of this mean distribution is slightly shifted to the right.

* Also, the variance of the samples; whether we can say qualitatively that this is acceptable or not depends on the problem. In general, the more representative data points available to us, the lower the variance.

Another thing to note here is that we modeled this problem using a Gaussian distribution, however we have some outliers that need to be accounted for which we cannot do well with a Gaussian distribution. We will see below how to use a Student's t-distribution for that.

y_pred_g = pm.sample_posterior_predictive(trace_g, 100, model_g)
print("Shape of the sampled variable y and data ",np.shape(y_pred_g['y']), len(data))

y_pred_g['y'][0]

data_ppc = az.from_pymc3(trace=trace_g, posterior_predictive=y_pred_g)
ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=True)
ax[0].legend(fontsize=15)

<br>
<br>
<hr style="border:2px solid blue"> </hr>

### Robust Models with a Student's t-Distribution

As mentioned in the previous section, one of the issues with assuming a Gaussian distribution is the assumption of finite variance. When you have observed data that lies outside this 'boundary', a Gaussian distribution is not a good fit and PyMC3, and other MCMC-based tools will be unable to reconcile these differences appropriately.

The probability density function is given by:

$P(t) = \dfrac{\gamma ((v+1) / 2)}{\sqrt{v \pi} \gamma (v/2)} (1 + \dfrac{t^2}{v})^{-(v+1)/2}$

μ corresponds to the mean of the distribution
    
σ is the scale and corresponds roughly to the standard deviation

ν is the degrees of freedom and takes values between 0 and $\infty$. The degrees of freedom corresponds to the number of independent observations minus 1. When the sample size is 8, the t-distribution used to model this would have degrees of freedom set to 9. A value of 1 corresponds to the Cauchy distribution and indicates heavy tails, while infinity corresponds to a Normal distribution. 

The mean of the distribution is 0 and the variance is given by ν/(ν - 2).

Now let us model the same problem with this distribution instead of a Normal.


with pm.Model() as model_t:
    μ = pm.Uniform('μ', 40, 75) # mean
    σ = pm.HalfNormal('σ', sd=10)
    ν = pm.Exponential('ν', 1/30)
    y = pm.StudentT('y', mu=μ, sd=σ, nu=ν, observed=data)
    trace_t = pm.sample(1000)

az.plot_trace(trace_t)
pm.model_to_graphviz(model_t)    

Using a student's t-distribution we notice that the outliers are captured more accurately now and the model fits better.

# Using a student's t distribution we notice that the outliers are captured more 
# accurately now and the model fits better
y_ppc_t = pm.sample_posterior_predictive(
    trace_t, 100, model_t, random_seed=123)
y_pred_t = az.from_pymc3(trace=trace_t, posterior_predictive=y_ppc_t)
az.plot_ppc(y_pred_t, figsize=(12, 6), mean=True)
ax[0].legend(fontsize=15)
plt.xlim(40, 70)

### Reading - Bayesian Estimation to Determine the Effectiveness of Drugs 

https://docs.pymc.io/notebooks/BEST.html

<br>
<br>
<hr style="border:2px solid blue"> </hr>

### Hierarchical Models or Multilevel Models

Suppose we want to perform an analysis of water quality in a state and we divide this state into districts with information available from each district, there are two options to do this. 

* We can study each district separately, however we lose information especially if there is insufficient data for some districts. But we get a more detailed model per district.

* The second option is to combine all the data and estimate the water quality of the state as a whole, i.e. a pooled model. We have more data but we lose granular information about each district.

The hierarchical model combines both of these options, by sharing information between the districts using hyperpriors that are priors over the parameter priors. In other words, instead of setting the parameter priors to a constant value, we draw it from another prior distribution called the hyperprior. This hyperprior is shared among all the districts and as a result sharing information between all the groups in the data.

#### Problem Statement

We measure the water samples for three districts, and we collect 30 samples for each district. The data is simply a binary value that indicates whether the water is contaminated or not. We count the number of samples that have contamination below the acceptable levels. We generate three arrays:

* N_samples - The total number of samples collected for each district or group
* G_samples - The number of good samples or samples with contamination levels below a certain threshold
* group_idx - The id for each district or group

#### Artifically generate the data

N_samples = [30, 30, 30] # Total number of samples collected
G_samples = [18, 18, 18] # Number of samples with water contamination 
                         # below accepted levels
# Create an id for each of the 30 + 30 + 30 samples - 0,1,2 to indicate that they
# belong to different groups
group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))


group_idx

data

#### The Sampling Model

The scenario presented here is essentially a binary classification problem that can be modeled using a Bernoulli distribution. The parameter of the Bernoulli distribution is a vector corresponding to each group (\\(\theta_1, \theta_2, \theta_3\\)) and indicates the probability of getting a good sample (in each group). Since this is a hierarchical model, each group shares information and a result the parameter of Group 1 can be influenced by the samples in Group 2 and 3. This is what makes hierarchical modeling so powerful. 

The process of generating our samples looks like the following. If we start from the last equation and work our way up, we can see that $\theta_i$ and $y_i$ are similar to a pooled model except that the *Beta* prior takes parameters $\alpha$ and $\beta$ instead of constant values. These parameters now have hyperpriors applied to them using the parameters $\mu$ and *k* which are assumed to be distributed using a *Beta* distribution and a half-Normal distribution respectively. Note that $\alpha$ and $\beta$ are indirectly computed from the terms \\(\mu\\) and *k* here. \\(\mu\\) affects the mean of the Beta distribution and increasing *k* makes the Beta distribution mor concentrated. This parameterization is more efficient than the direct parameterization in terms of $\alpha_i$ and $\beta_i$.


$ \mu \sim Beta(\alpha_p, \beta_p)  \\
  k \sim | Normal(0,\sigma_k) | \\
  \alpha =  \mu * k \\
  \beta = (1 - \mu) * k \\
  \theta_i \sim Beta(\alpha, \beta) \\
  y_i \sim Bern(\theta_i)
$



def get_hyperprior_model(data, N_samples, group_idx):
    with pm.Model() as model_h:
        μ = pm.Beta('μ', 1., 1.) # hyperprior
        κ = pm.HalfNormal('κ', 10) # hyperprior
        alpha = pm.Deterministic('alpha', μ*κ)
        beta = pm.Deterministic('beta', (1.0-μ)*κ)
        θ = pm.Beta('θ', alpha=alpha, beta=beta, shape=len(N_samples)) # prior, len(N_samples) = 3
        y = pm.Bernoulli('y', p=θ[group_idx], observed=data)
        trace_h = pm.sample(2000)
    az.plot_trace(trace_h)
    print(az.summary(trace_h))
    return(model_h)    
    
model = get_hyperprior_model(data, N_samples, group_idx)
pm.model_to_graphviz(model)

#### Shrinkage

Shrinkage refers to the phenomenon of sharing information among the groups through the use of hyperpriors. Hierarchical models can therefore be considered partially pooled models since information is shared among the groups so we move away from extremes, which is great if we have outliers (or poor quality data) in our data groups. This is particularly useful if  we do not have a lot of data. Here, the groups are neither independent nor do we clump all the data together without accounting for the differences in the groups. This can 

We can look at three cases below as examples to illustrate what happens here. We keep the total number of samples and the groups the same as before, however we vary the number of good samples in each group. When there are significant differences in the number of good sample between groups, the behavior is different from what we see with an independent model, averages win and extreme values are avoided.

The values of G_samples are changed to have the following values

1. [5,5,5]
2. [18,5,5]
3. [18,18,1]

Note how the values of the three $\theta$s change as we change the values of G_samples.

# Case 1
N_samples = [30, 30, 30] # Total number of samples collected
G_samples = [5, 5, 5] # Number of samples with water contamination 
                         # below accepted levels
# Create an id for each of the 30 + 30 + 30 samples - 0,1,2 to indicate that they
# belong to different groups
group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))

model = get_hyperprior_model(data, N_samples, group_idx)
model

# Case 2 - The value of theta_1 is now smaller compared to our original case
N_samples = [30, 30, 30] # Total number of samples collected
G_samples = [18, 5, 5] # Number of samples with water contamination 
                         # below accepted levels
# Create an id for each of the 30 + 30 + 30 samples - 0,1,2 to indicate that they
# belong to different groups
group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))

get_hyperprior_model(data, N_samples, group_idx)

# Case 3 - Value of theta_3 is not as small as it would have been if it were estimated individually
N_samples = [30, 30, 30] # Total number of samples collected
G_samples = [18, 18, 1] # Number of samples with water contamination 
                         # below accepted levels
# Create an id for each of the 30 + 30 + 30 samples - 0,1,2 to indicate that they
# belong to different groups
group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))

get_hyperprior_model(data, N_samples, group_idx)

### GRADED EVALUATION (18 mins)

1. According to the Central Limit Theorem, the mean of the sample means tends to the true population mean as the number of samples increase

    a. True (C)

    b. False
    
    
2. Many real-world phenomena are averages of various factors, hence it is reasonable to use a Gaussian distribution to model them

    a. True (C)
    
    b. False 
    
    
3. What type of distribution is better suited to modeling positive values?

    a. Normal
    
    b. Half-normal (C)
    
    
4. Posterior predictive checks can be used to verify that the inferred distribution is similar to the observed data

    a. True (C)
    
    b. False
    
    
5. Which distribution is better suited to model data that has a lot of outliers?

    a. Gaussian distribution
    
    b. Student's t-distribution (C)
    
    
6. Hierarchical models are beneficial in modeling data from groups where there might be limited data in certain groups

    a. True (C)
    
    b. False
    
  
7. Hierarchical models share information through hyperpriors

    a. True (C)
    
    b. False

<br>
<br>
<hr style="border:2px solid blue"> </hr>


### Linear Regression Again!

Let us generate some data and plot it along with its density.

np.random.seed(1)
N = 100
alpha_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)
x = np.random.normal(10, 1, N)
y_real = alpha_real + beta_real * x
y = y_real + eps_real
_, ax = plt.subplots(1,2, figsize=(8, 4))
ax[0].plot(x, y, 'C0.')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y', rotation=0)
ax[0].plot(x, y_real, 'k')
az.plot_kde(y, ax=ax[1])
ax[1].set_xlabel('y')
plt.tight_layout()


#### Inference of Parameters in Linear Regression

import pymc3 as pm

with pm.Model() as model_g:
    α = pm.Normal('α', mu=0, sd=10)
    β = pm.Normal('β', mu=0, sd=1)
    ϵ = pm.HalfCauchy('ϵ', 5)
    μ = pm.Deterministic('μ', α + β * x)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ϵ, observed=y)
    trace_g = pm.sample(2000, tune=1000)
    
az.plot_trace(trace_g, var_names=['α', 'β', 'ϵ'])
plt.figure()


#### Parameter Correlations

az.plot_pair(trace_g, var_names=['α', 'β'], plot_kwargs={'alpha': 0.1})

plt.figure()
plt.plot(x, y, 'C0.') # Plot the true values
alpha_m = trace_g['α'].mean() # Mean of the inferred values
beta_m = trace_g['β'].mean()  # Mean of the inferred values
draws = range(0, len(trace_g['α']), 10)
plt.plot(x, trace_g['α'][draws] + trace_g['β'][draws]* x[:, np.newaxis], c='lightblue', alpha=0.5)
plt.plot(x, alpha_m + beta_m * x, c='teal', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()

#### We sample from the posterior

ppc = pm.sample_posterior_predictive(trace_g,
                                     samples=2000,
                                     model=model_g)

plt.plot(x, y, 'b.') # Plot the true y values
plt.plot(x, alpha_m + beta_m * x, c='teal') # Plot the mean
az.plot_hpd(x, ppc['y_pred'], credible_interval=0.5, color='lightblue')

#### Mean center the data

Looking at the plot of $\alpha$ and $\beta$, one can notice the high degree of correlation between these two variables. This results in a parameter posterior space that is diagonally shaped, which is problematic for many samplers such as the Metropolis-Hastings MCMC sampler. One recommended approach to minimize this correlation is to center the independent variables.

$\tilde{x} = x - \bar{x}$

The advantage of this is twofold

1. The pivot point is the intercept when the slope changes
2. The parameter posterior space is more circular

##### Transformation

In order to center the data

$$y = \alpha - \beta x$$
can be reformulated as 

$$y = \alpha - \tilde{\beta}(x - \bar{x}) = \alpha + \tilde{\beta} \bar{x} - \tilde{\beta} x$$

$$y = \tilde{\alpha} - \tilde{\beta} x$$

##### Recovering the data

This implies that the original alpha can now be recovered using the formula

$$\alpha = \tilde{\alpha} - \tilde{\beta} \bar{x}$$

You can also standardize the data by mean centering and dividing by the standard deviation

$$\tilde{x} = (x - \bar{x}) / \sigma_x$$

#### Mean Centered - Broader Sampling Space

# Center the data
x_centered = x - x.mean()

with pm.Model() as model_g:
    α = pm.Normal('α', mu=0, sd=10)
    β = pm.Normal('β', mu=0, sd=1)
    ϵ = pm.HalfCauchy('ϵ', 5)
    μ = pm.Deterministic('μ', α + β * x_centered)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ϵ, observed=y)
    α_recovered = pm.Deterministic('α_recovered', α - β * x.mean())
    trace_g = pm.sample(2000, tune=1000)
    
az.plot_trace(trace_g, var_names=['α', 'β', 'ϵ'])
plt.figure()
az.plot_pair(trace_g, var_names=['α', 'β'], plot_kwargs={'alpha': 0.1})

### Robust Linear Regression

We fitted our model parameters by assuming the data likelihood was a Normal distribution, however as we saw earlier this assumption suffers from not doing well with outliers. Our solution to this problem is the same, use a Student's t-distribution for the likelihood.

Here we look at the [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet), which is a set of 4 data sets. They have similar statistical properties even though they look very different and were used to illustrate the effect of outliers. Our intended goal is the same, how to model data with outliers and the sensitivity of the model to these outliers.

import seaborn as sns
from scipy import stats

# Load the example dataset for Anscombe's quartet
df = sns.load_dataset("anscombe")
df

#### Plot the 4 Subgroups in the Data

x_0 = df[df.dataset == 'I']['x'].values
y_0 = df[df.dataset == 'I']['y'].values
x_1 = df[df.dataset == 'II']['x'].values
y_1 = df[df.dataset == 'II']['y'].values
x_2 = df[df.dataset == 'III']['x'].values
y_2 = df[df.dataset == 'III']['y'].values
x_3 = df[df.dataset == 'IV']['x'].values
y_3 = df[df.dataset == 'IV']['y'].values
_, ax = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=True)

ax = np.ravel(ax)
ax[0].scatter(x_0, y_0)
ax[0].set_title('Group I')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y', rotation=0, labelpad=15)
ax[1].scatter(x_1, y_1)
ax[1].set_title('Group II')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y', rotation=0, labelpad=15)
ax[2].scatter(x_2, y_2)
ax[2].set_title('Group III')
ax[2].set_xlabel('x')
ax[2].set_ylabel('y', rotation=0, labelpad=15)
ax[3].scatter(x_3, y_3)
ax[3].set_title('Group IV')
ax[3].set_xlabel('x')
ax[3].set_ylabel('y', rotation=0, labelpad=15)

#### Plot Data Group 3 and its Kernel Density

x_2 = x_2 - x_2.mean()
_, ax = plt.subplots(1, 2, figsize=(10, 5))
beta_c, alpha_c = stats.linregress(x_2, y_2)[:2]
ax[0].plot(x_2, (alpha_c + beta_c * x_2), 'k',
           label=f'y ={alpha_c:.2f} + {beta_c:.2f} * x')
ax[0].plot(x_2, y_2, 'C0o')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y', rotation=0)
ax[0].legend(loc=0)
az.plot_kde(y_2, ax=ax[1], rug=True)
ax[1].set_xlabel('y')
ax[1].set_yticks([])
plt.tight_layout()

#### Model using a Student's t Distribution

with pm.Model() as model_t:
    α = pm.Normal('α', mu=y_2.mean(), sd=1)
    β = pm.Normal('β', mu=0, sd=1)
    ϵ = pm.HalfNormal('ϵ', 5)
    ν_ = pm.Exponential('ν_', 1/29)
    ν = pm.Deterministic('ν', ν_ + 1)
    y_pred = pm.StudentT('y_pred', mu=α + β * x_2,
                         sd=ϵ, nu=ν, observed=y_2)
    trace_t = pm.sample(2000)
    
alpha_m = trace_t['α'].mean()
beta_m = trace_t['β'].mean()
plt.plot(x_2, alpha_m + beta_m * x_2, c='k', label='robust')
plt.plot(x_2, y_2, '*')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend(loc=2)
plt.tight_layout()

pm.model_to_graphviz(model_t)

### Hierarchical Linear Regression

We want to use the same hierarchical or multilevel modeling technique for linear regression problems. As mentioned above, this is particularly useful when presented with imbalanced subgroups of sparse data. In this example, we create 8 subgroups with 7 of them having 20 data points and the last one having a single data point. 

The data for the 8 groups are generated from a Normal distribution of mean 10 and a standard deviation of 1. The parameters for the linear model are generated from the Normal and Beta distributions.

#### Data Generation

N = 20
M = 8
idx = np.repeat(range(M-1), N)
idx = np.append(idx, 7)

np.random.seed(314)
alpha_real = np.random.normal(4, 1, size=M) 
beta_real = np.random.beta(7, 1, size=M)
eps_real = np.random.normal(0, 0.5, size=len(idx))
print("Alpha parameters ", alpha_real )

y_m = np.zeros(len(idx))
x_m = np.random.normal(10, 1, len(idx))
y_m = alpha_real[idx] + beta_real[idx] * x_m + eps_real
_, ax = plt.subplots(2, 4, figsize=(12,8), sharex=True, sharey=True)

ax = np.ravel(ax)
j, k = 0, N
for i in range(M):
    ax[i].scatter(x_m[j:k], y_m[j:k])
    ax[i].set_xlabel(f'x_{i}')
    ax[i].set_ylabel(f'y_{i}', rotation=0, labelpad=15)
    ax[i].set_xlim(6, 15)
    ax[i].set_ylim(7, 17)     
    j += N
    k += N
plt.tight_layout()

#### Non-hierarchical Model

We build a non-hierarchical model first for comparison. We also mean-center the data for ease of convergence. Note how the obtained $\alpha$ and $\beta$ values vary for each group, particularly the scale of the last one, which is really off.

# Center the data
x_centered = x_m - x_m.mean()

with pm.Model() as unpooled_model:
    # Note the M prior parameters for the M groups
    α_tmp = pm.Normal('α_tmp', mu=2, sd=5, shape=M)
    β = pm.Normal('β', mu=0, sd=10, shape=M)
    ϵ = pm.HalfCauchy('ϵ', 5)
    ν = pm.Exponential('ν', 1/30)
    
    y_pred = pm.StudentT('y_pred', mu=α_tmp[idx] + β[idx] * x_centered, sd=ϵ, nu=ν, observed=y_m)
    # Rescale alpha back - after x had been centered the computed alpha is different from the original alpha
    α = pm.Deterministic('α', α_tmp - β * x_m.mean())
    trace_up = pm.sample(2000)

az.plot_trace(trace_up)
plt.figure()
az.plot_forest(trace_up, var_names=['α', 'β'], combined=True)
az.summary(trace_up)

pm.model_to_graphviz(unpooled_model)

#### Hierarchical Model

We set hyperpriors on the \\(\alpha\\) and \\(\beta\\) parameters. To be more precise, the hyperpriors are applied to the scaled version of \\(\alpha\\), i.e. \\(\alpha_{tmp}\\).

with pm.Model() as hierarchical_model:
    # Hyperpriors - we add these instead of setting the prior values to a constant
    # Note that there exists only one hyperprior  for all M groups, shared hyperprior
    α_μ_tmp = pm.Normal('α_μ_tmp', mu=100, sd=1) # try changing these hyperparameters
    α_σ_tmp = pm.HalfNormal('α_σ_tmp', 10) # try changing these hyperparameters
    β_μ = pm.Normal('β_μ', mu=10, sd=2) # reasonable changes do not have an impact
    β_σ = pm.HalfNormal('β_σ', sd=5)
    
    # priors - note that the prior parameters are no longer a constant
    α_tmp = pm.Normal('α_tmp', mu=α_μ_tmp, sd=α_σ_tmp, shape=M)
    β = pm.Normal('β', mu=β_μ, sd=β_σ, shape=M)
    ϵ = pm.HalfCauchy('ϵ', 5)
    ν = pm.Exponential('ν', 1/30)
    y_pred = pm.StudentT('y_pred',
                         mu=α_tmp[idx] + β[idx] * x_centered,
                         sd=ϵ, nu=ν, observed=y_m)
    α = pm.Deterministic('α', α_tmp - β * x_m.mean())
    α_μ = pm.Deterministic('α_μ', α_μ_tmp - β_μ *
                           x_m.mean())
    α_σ = pm.Deterministic('α_sd', α_σ_tmp - β_μ * x_m.mean())
    trace_hm = pm.sample(1000)

az.plot_forest(trace_hm, var_names=['α', 'β'], combined=True)

_, ax = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True,
                     constrained_layout=True)
ax = np.ravel(ax)
j, k = 0, N
x_range = np.linspace(x_m.min(), x_m.max(), 10)
for i in range(M):
    ax[i].scatter(x_m[j:k], y_m[j:k])
    ax[i].set_xlabel(f'x_{i}')
    ax[i].set_ylabel(f'y_{i}', labelpad=17, rotation=0)
    alpha_m = trace_hm['α'][:, i].mean()
    beta_m = trace_hm['β'][:, i].mean()
    ax[i].plot(x_range, alpha_m + beta_m * x_range, c='k',
               label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
    plt.xlim(x_m.min()-1, x_m.max()+1)
    plt.ylim(y_m.min()-1, y_m.max()+1)
    j += N
    k += N

pm.model_to_graphviz(hierarchical_model)

<br>
<br>
<hr style="border:2px solid blue"> </hr>

### Polynomial Regression for Nonlinear Data

What happens when the data is inherently nonlinear? It is more appropriate to use non-linear combinations of the inputs. This could be in the form of higher order terms such as \\(x^2, x^3\\) or it could use basis functions such as the cosine function, \\(cos(x)\\).

#### Data Generation

Use the values from the dataset in Anscombe's quartet we used earlier as our non-linear data. We will use the regression model given by

$$ y = \alpha + \beta_1 * x_{centered} + \beta_2 * x_{centered}^2 $$

x_1_centered = x_1 - x_1.mean()
plt.scatter(x_1_centered, y_1)
plt.xlabel('x')
plt.ylabel('y', rotation=0)

plt.figure()
x_0_centered = x_0 - x_0.mean()
plt.scatter(x_0_centered, y_0)
plt.xlabel('x')
plt.ylabel('y', rotation=0)

#### Inference on Data I

with pm.Model() as model_poly:
    α = pm.Normal('α', mu=y_1.mean(), sd=1)
    β1 = pm.Normal('β1', mu=0, sd=1)
    β2 = pm.Normal('β2', mu=0, sd=1)
    ϵ = pm.HalfCauchy('ϵ', 5)
    mu = α + β1 * x_1_centered + β2 * x_1_centered**2
    y_pred = pm.Normal('y_pred', mu=mu, sd=ϵ, observed=y_1)
    trace = pm.sample(2000, tune=2000)
    
x_p = np.linspace(-6, 6)
y_p = trace['α'].mean() + trace['β1'].mean() * x_p + trace['β2'].mean() * x_p**2
plt.scatter(x_1_centered, y_1)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.plot(x_p, y_p, c='C1')



#### Inference on Data II

with pm.Model() as model_poly:
    α = pm.Normal('α', mu=y_0.mean(), sd=1)
    β1 = pm.Normal('β1', mu=0, sd=1)
    β2 = pm.Normal('β2', mu=0, sd=1)
    ϵ = pm.HalfCauchy('ϵ', 5)
    mu = α + β1 * x_0_centered + β2 * x_0_centered**2
    y_pred = pm.Normal('y_pred', mu=mu, sd=ϵ, observed=y_0)
    trace = pm.sample(2000)
    
x_p = np.linspace(-6, 6)
y_p = trace['α'].mean() + trace['β1'].mean() * x_p + trace['β2'].mean() * x_p**2
plt.scatter(x_0_centered, y_0)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.plot(x_p, y_p, c='C1')

<br>
<br>
<hr style="border:2px solid blue"> </hr>

### Multiple Linear Regression

In Multiple Linear Regression, there is more than one independent variable to predict the outcome of one dependent variable.

$$ y = \alpha + \overrightarrow{\beta} \cdot \overrightarrow{X} $$

#### Data Generation

The example below generates two-dimensional data for X. It plots the variation of 'y' with each component of X in the top two figures. The bottom figure indicates the correlation of the two components of X.

np.random.seed(314)

# N is the total number of observations 
N = 100
# m is 2, the number of independent variables
alpha_real = 2.5
beta_real = [0.9, 1.5]
eps_real = np.random.normal(0, 0.5, size=N)

# X is # n x m
X = np.array([np.random.normal(i, j, N) for i, j in zip([10, 2], [1, 1.5])]).T 
X_mean = X.mean(axis=0, keepdims=True)
X_centered = X - X_mean
y = alpha_real + np.dot(X, beta_real) + eps_real

def scatter_plot(x, y):
    plt.figure(figsize=(10, 10))
    for idx, x_i in enumerate(x.T):
        plt.subplot(2, 2, idx+1)
        plt.scatter(x_i, y)
        plt.xlabel(f'x_{idx+1}')
        plt.ylabel(f'y', rotation=0)
    plt.subplot(2, 2, idx+2)
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlabel(f'x_{idx}')
    plt.ylabel(f'x_{idx+1}', rotation=0)

scatter_plot(X_centered, y)

#### Inference

This code is very similar to what we have already seen, the only real difference being the dimensionality of the coefficients and the inputs. Something you would notice is that as the number of unknowns increase, the uncertainty associated with our inferences become larger. It is beneficial to have more accurate priors in this situation.

with pm.Model() as model_mlr:
    α_tmp = pm.Normal('α_tmp', mu=2, sd=2) # Try changing the prior distribution
    β = pm.Normal('β', mu=0, sd=5, shape=2) # Note the shape of beta
    ϵ = pm.HalfCauchy('ϵ', 5)
    μ = α_tmp + pm.math.dot(X_centered, β)
    α = pm.Deterministic('α', α_tmp - pm.math.dot(X_mean, β))
    y_pred = pm.Normal('y_pred', mu=μ, sd=ϵ, observed=y)
    trace = pm.sample(2000, tune=1000)
    
az.summary(trace)

pm.model_to_graphviz(model_mlr)

<br>
<br>
<hr style="border:2px solid blue"> </hr>

### Logistic Regression

While everything we have seen so far involved regression, the same ideas can be applied to a classification task as well. We use the logistic regression model to perform this classification here. The name 'regression' is due to the fact that the model outputs class probabilities as numbers which is then converted into classes using a decision boundary. There are many ways to select an appropriate decision boundary, a few of which were covered in Course 1 and Course 2.

#### Inverse Link function

At this point it is a good idea to bring up the concept of a inverse link function, which takes the form

$\theta = f(\alpha + \beta x)$

Here 'f' is called the inverse link function, the term inverse refers to the fact that the function is applied to the right hand side of the equation. In a linear regression, this inverse link function is the identity function. In the case of a linear regression model, the value 'y' at any point 'x' is modeled as the mean of a Gaussian distribution centered at the point (x,y). The error as a result of the true 'y' and the estimated 'y' are modeled with the standard deviation of this Gaussian at that point (x,y). Now think about the scenario where this is not appropriately modeled using a Gaussian. A classification problem is a perfect example of such a scenario where the discrete classes are not modeled well as a Gaussian and hence we can't use this distribution to model the mean of those classes. As a result, we would like to convert the output of $\alpha + \beta x$ to some other range of values that are more appropriate to the problem being modeled, which is what the link function intends to do.

#### Logistic function

The logistic function is defined as the function

$logistic(x) = \dfrac{1}{1 + \exp{(-x)}}$

This is also called the sigmoid function and it restricts the value of the output to the range [0,1].

x = np.linspace(-5,5)
plt.plot(x, 1 / (1 + np.exp(-x)))
plt.xlabel('x')
plt.ylabel('logistic(x)')


#### Example using the Iris data

The simplest example using a logistic regression model is one that can be used to identify two classes. If you are given a set of independent variables that are features which correspond to an output dependent variable that is a class, you can build a model to learn the relationship between the features and the output classes. This is done with the help of the logistic function which acts as the inverse link function to relate the features to the output class. 

$\theta = logistic(\alpha + \beta x)$

If it is a two-class problem (binary classification), the output variable can be represented by a Bernoulli distribution.

$y \sim Bern(\theta)$

The mean parameter $\theta$ is now given by the regression equation $logistic(\alpha + \beta x)$. In regular linear regression, this parameter was drawn from a Gaussian distribution. In the case of the coin-flip example the data likelihood was represented by a Bernoulli distribution, (the parameter $\theta$ was drawn from a Beta prior distribution there), similarly we have output classes associated with every observation here.

We load the iris data from scikit learn and

* Plot the distribution of the three classes for two of the features. 

* We also perform a pairplot to visualize the correlation of each feature with every other feature. The diagonal of this plot shows the distribution of the three classes for that feature.

* Correlation plot of just the features. This can be visually cleaner and cognitively simpler to comprehend.

import pymc3 as pm
import sklearn
import numpy as np
import graphviz
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
from sklearn import datasets
df = datasets.load_iris()
iris_data = pd.DataFrame(df['data'], columns=df['feature_names'])
iris_data['target'] = df['target']
seaborn.stripplot(x='target', y='sepal length (cm)', data=iris_data, jitter=False)
plt.figure()
seaborn.stripplot(x='target', y='petal length (cm)', data=iris_data, jitter=False)
plt.figure()
seaborn.pairplot(iris_data, hue='target', diag_kind='kde')
plt.figure()
corr = iris_data.query("target == (0,1)").loc[:, iris_data.columns != 'target'].corr() 
mask = np.tri(*corr.shape).T 
seaborn.heatmap(corr.abs(), mask=mask, annot=True)
plt.show()

You would notice that some of the variables have a high degree of correlation from the correlation plot. One approach is to eliminate one of the correlated variables. The second option is to mean-center and use a weakly-informative prior such as a Students t-distribution for all variables that are not binary. The scale parameter can be adjusted for the range of expected values for these variables and the normality parameter is recommended to be between 3 and 7. (Source: Andrew Gelman and the Stan team)



df['target_names']

#### Inference

We use a **single feature**, the sepal length, to learn a decision boundary between the **first two classes** in the iris data (0,1).

In this case, the decision boundary is defined to be the value of 'x' when 'y' = 0.5. We won't go over the derivation here, but this turns out to be \\(-\alpha / \beta\\). However, this value was chosen under the assumption that the midpoint of the class values are a good candidate for separating the classes, but this does not have to be the case.

# Select the first two classes for a binary classification problem
df = iris_data.query("target == (0,1)")
y_0 = df.target
x_n = 'sepal length (cm)' 
x_0 = df[x_n].values
x_c = x_0 - x_0.mean()

import pymc3 as pm
import arviz as az

with pm.Model() as model_0:
    α = pm.Normal('α', mu=0, sd=10)
    β = pm.Normal('β', mu=0, sd=10)
    μ = α + pm.math.dot(x_c, β)    
    θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
    bd = pm.Deterministic('bd', -α/β)
    yl = pm.Bernoulli('yl', p=θ, observed=y_0)
    trace_0 = pm.sample(1000)

pm.model_to_graphviz(model_0)

az.summary(trace_0, var_names=["α","β","bd"])

#### Visualizing the Decision Boundary

* The classifier outputs, i.e. the y values are jittered to make it easier to visualize. 
* The solid green lines are the mean of the fitted \\(\theta\\) as a result of the sampling and inference process
* The transparent green lines indicate the 94% HPD (default values) for the fitted \\(\theta\\).
* The solid blue line indicates the decision boundary that is derived from the inferred values of the parameters using the equation \\(\alpha/\beta\\)
* The transparent blue indicates the HPD (94%) for the decision boundary.

theta = trace_0['θ'].mean(axis=0)
idx = np.argsort(x_c)

# Plot the fitted theta
plt.plot(x_c[idx], theta[idx], color='teal', lw=3)
# Plot the HPD for the fitted theta
az.plot_hpd(x_c, trace_0['θ'], color='teal')
plt.xlabel(x_n)
plt.ylabel('θ', rotation=0)

# Plot the decision boundary
plt.vlines(trace_0['bd'].mean(), 0, 1, color='steelblue')
# Plot the HPD for the decision boundary
bd_hpd = az.hpd(trace_0['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='steelblue', alpha=0.5)
plt.scatter(x_c, np.random.normal(y_0, 0.02),
            marker='.', color=[f'C{x}' for x in y_0])

# use original scale for xticks
locs, _ = plt.xticks()
plt.xticks(locs, np.round(locs + x_0.mean(), 1))

#### Multiple Logistic Regression

The above example with a single feature can be extended to take **multiple features** or independent variables to separate the same **two classes**.

# Select the first two classes for a binary classification problem
df = iris_data.query("target == (0,1)")
y_0 = df.target
x_n = ['sepal length (cm)', 'sepal width (cm)']
# Center the data by subtracting the mean from both columns
df_c = df - df.mean() 
x_c = df_c[x_n].values


As we saw before, the equation for multiple logistic regression relating the $\theta$ parameter to the features can be written as 

$$\theta = logistic(\alpha + \beta_1 x_1 + \beta_2 x_2)$$

$$y \sim Bern(\theta)$$

This gives us a decision boundary, assuming y = 0.5 is a reasonable boundary, of 

$$x_2 = -\dfrac{\alpha}{\beta_2} - \dfrac{\beta_1}{\beta_2} x_1$$

Unlike the previous equation, this one represents a line for the variables \\(x_1\\) and \\(x_2\\) which separates the two-dimensional space occupied by \\(x_1\\) and \\(x_2\\). For higher dimensions, this boundary decision will be a hyperplane of dimension 'n-1' for a feature space of dimension 'n'.

#### Inference

with pm.Model() as model_1: 
    α = pm.Normal('α', mu=0, sd=10) 
    β = pm.Normal('β', mu=0, sd=2, shape=len(x_n)) 
    μ = α + pm.math.dot(x_c, β) 
    θ = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-μ))) 
    bd = pm.Deterministic('bd', -α/β[1] - β[0]/β[1] * x_c[:,0])
    yl = pm.Bernoulli('yl', p=θ, observed=y_0) 
    trace_0 = pm.sample(2000)


#### Visualization

We plot the HPD on the centered data, we have not scaled it back to the original range here.

idx = np.argsort(x_c[:,0]) 
bd = trace_0['bd'].mean(0)[idx] 
plt.scatter(x_c[:,0], x_c[:,1], c=[f'C{x}' for x in y_0]) 
plt.plot(x_c[:,0][idx], bd, color='steelblue'); 
az.plot_hpd(x_c[:,0], trace_0['bd'], color='steelblue')
plt.xlabel(x_n[0]) 
plt.ylabel(x_n[1])

pm.model_to_graphviz(model_1)

### Multiclass Classification

If we have more than two classes this becomes a multiclass problem. In this case we use the softmax function instead of the sigmoid function. The sigmoid function is a special case of the softmax for a two-class classification problem. The softmax function can be written as

$$ softmax(x_i) = \dfrac{\exp(x_i)}{\sum_k \exp(x_k)} $$

Earlier, we also used a Bernoulli distribution as the likelihood for our $\theta$ parameter, however now we sample from a categorical distribution.

$$\theta = logistic(\alpha + \beta x)$$

$$y \sim Categorical(\theta)$$

#### Logistic Regression for a Multiclass Problem 

We reuse the previous example, however we are going to use all the three classes in the data here along with all the features for maximum separability. The data is also standardized instead of just mean-centered here.


%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import arviz as az
import pymc3 as pm
import numpy as np
import graphviz
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import datasets
df = datasets.load_iris()
iris_data = pd.DataFrame(df['data'], columns=df['feature_names'])
iris_data['target'] = df['target']
y_s = iris_data.target
x_n = iris_data.columns[:-1]
x_s = iris_data[x_n]
x_s = (x_s - x_s.mean()) / x_s.std()
x_s = x_s.values

import theano as tt
tt.config.gcc.cxxflags = "-Wno-c++11-narrowing"

with pm.Model() as model_mclass:
    alpha = pm.Normal('alpha', mu=0, sd=5, shape=3)
    beta = pm.Normal('beta', mu=0, sd=5, shape=(4,3))
    μ = pm.Deterministic('μ', alpha + pm.math.dot(x_s, beta))
    θ = tt.tensor.nnet.softmax(μ)
    #θ = pm.math.exp(μ)/pm.math.sum(pm.math.exp(μ), axis=0)
    yl = pm.Categorical('yl', p=θ, observed=y_s)
    trace_s = pm.sample(2000)

data_pred = trace_s['μ'].mean(0)
y_pred = [np.exp(point)/np.sum(np.exp(point), axis=0) for point in data_pred]
az.plot_trace(trace_s, var_names=['alpha'])
f'{np.sum(y_s == np.argmax(y_pred, axis=1)) / len(y_s):.2f}'


<br>
<br>
<hr style="border:2px solid blue"> </hr>

### Inferring Rate Change with a Poisson Distribution

Discrete variables that represents count data can be handled using a Poisson distribution. The key element here is that it is the number of events happening in a given interval of time. The events are supposed to be independent and   the distribution is parameterized using a single value called the rate parameter. This corresponds to and controls both the mean and the variance of the distribution. One implication of this is that the higher the mean, the larger the variance of the distribution which can be a limitation for some phenomena. A higher value of the rate parameter indicates a higher likelihood of getting larger values from our distribution. It is represented by

$f(x) = e^{-\mu} \mu^x / x!$

* The mean rate is represented by $\mu$
* x is a positive integer that represents the number of events that can happen

If you recall from the discussion of the binomial distribution, that can also be used to model the probability of the number of successes out of 'n' trials. The Poisson distribution is a special case of this binomial distribution and is used when the trials far exceed the number of successes. 

#### Poisson Distribution Example

In the following example we look at a time-varying rate phenomena, consider the observations as the number of COVID-19 cases per day. We observe cases for 140 days, however due to some interventional measures put in place it is suspected that the number of cases per day have gone down. If we assume that the number of cases can be modeled using a Poisson distribution, then this implies that there are two rates $\lambda_1$ and $\lambda_2$, and we can try to find where this rate-switch happens (time $\tau$). 

We don't really know a lot about these rates, so we select a prior for both which can be from an Exponential, Gamma or Uniform distributions. Both the Exponential and Gamma distributions work better than the Uniform distribution since the Uniform distribution is the least informative. As usual, with enough observations one can even get away with a Uniform prior. Since we have no information regarding $\tau$, we select a Uniform prior distribution for that.

In the example below, try varying the following

1. Types of priors - a more informed prior is always better if this information is available
2. The size of the data or the observations and the value of the theta parameter - more data results in better inference overall, the larger the difference in theta the easier to determine these rates
3. The number of drawn samples - better and more accurate inference
4. The number of chains - should reduce variance
5. The number of cores - cores should be no more than the total number of chains and should be limited to the total number of cores on your hardware, you should see an increase in speed or decrease in runtime as you increase the number of cores.

Note that this uses the Metropolis algorithm since this is a discrete sampling problem.

# ------------ Create the data ---------- #
n_1 = 70
θ_real_1 = 7.5
#ψ = 0.1
 # Simulate some data
counts_1 = np.random.poisson(θ_real_1,n_1)
#plt.bar(np.arange(len(counts_1)),counts_1)

n_2 = 70
θ_real_2 = 2.0
#ψ = 0.1
 # Simulate some data
counts_2 = np.random.poisson(θ_real_2,n_2)
#plt.bar(np.arange(len(counts_2)),counts_2)

total_data = np.concatenate((counts_1, counts_2))
n_counts = len(counts_1) + len(counts_2)
plt.figure()
plt.bar(np.arange(len(total_data)),total_data)

# ------------ Generate the model ----------- #

with pm.Model() as model_poisson:

    alpha_1 = 1.0 / counts_1.mean()
    alpha_2 = 1.0 / counts_2.mean()

    # Different priors have different results                     
    lambda_1 = pm.Exponential("lambda_1", alpha_1)
    lambda_2 = pm.Exponential("lambda_2", alpha_2) 
    #lambda_1 = pm.Gamma("lambda_1", 2, 0.1)
    #lambda_2 = pm.Gamma("lambda_2", 2, 0.1)
    #lambda_1 = pm.Uniform("lambda_1",lower=0, upper=5)
    
    # Uniform prior for the day since we have no information, if we do we should modify the prior to         
    # incorporate that information
    tau = pm.DiscreteUniform("tau", lower=0, upper=n_counts - 1)
    idx = np.arange(n_counts) # id for the day
    lambda_c = pm.math.switch(tau > idx, lambda_1, lambda_2) # switch rate depending on the tau drawn

    observation = pm.Poisson("obs", lambda_c, observed=total_data)
    trace = pm.sample(5000, chains=10, cores=4)

az.plot_trace(trace)

#### Visualize \\(\tau\\)

We can use the ECDF (Empirical Cumulative Distribution Function) to visualize the distribution of \\(\tau\\). The ECDF helps to visualize the distribution by plotting the CDF as opposed to the binning techniques used by a histogram. It also helps us to identify what values in the data are beneath a certain probability.

It is expected that the rate parameter will be \\(\lambda_1\\) initially for a time till the day where the value of \\(\tau\\) has some probability mass 'x' when it switches to rate parameter \\(\lambda_2\\) with this probability mass 'x', whereas probability that this day could have the rate parameter \\(\lambda_1\\) will be '1 - x'. You could do this for all the days.

from statsmodels.distributions.empirical_distribution import ECDF
print('Tau is ',trace['tau'])
print("Length of tau", len(trace['tau']))
print('Lambda 1 is ',trace['lambda_1'])
print("Length of Lambda 1 ",len(trace['lambda_1']))
ecdf = ECDF(trace['tau'])
plt.plot(ecdf.x, ecdf.y, '-')

for elem in idx:
    prob_lambda_2 = ecdf([elem])
    prob_lambda_1 = 1.0 - prob_lambda_2
    print("Day %d, the probability of rate being lambda_1 is %lf and lambda_2 is %lf "%(elem, prob_lambda_1, prob_lambda_2))

#### Expected Value of  \\(\tau\\)

For each draw of $\tau$, there is a draw of $\lambda_1$ and $\lambda_2$. We can use the principles of Monte Carlo approximation to compute the expected value of COVID-19 cases on any day. 

Expected value for day = $ \dfrac{1}{N} \sum_{0}^{num\_samples}$ Lambda_draw ;  day > Tau_draw ? lambda_2_draw : lambda_1_draw

* Draws are in combinations of \\((\lambda_1, \lambda_2, \tau)\\), we want to average out the \\(\lambda\\) value based on the proportion of \\(\lambda\\) suggestions as indicated by the samples

* For days 0,...67 we see that the probability of \\(\lambda_1\\) is 1 whereas the probability of \\(\lambda_2\\) is 0. So the expected value is just the average of all the \\(\lambda_1\\) samples.

* Similarly, for days from 72,... the probability of \\(\lambda_2\\) is 1 and the probability of \\(\lambda_1\\) is 0. So the expected value of \\(\lambda\\) is just the average of all the \\(\lambda_2\\) samples.

* For days in between - let us assume for day 69, we have 10% of the samples indicating that \\(\tau\\) is 70 while 90% indicate that \\(\tau\\) is 69. 

 * If \\(\tau\\) is 70, that means that day 69 has rate \\(\lambda_1\\) but if \\(\tau\\) is 69 that implies that day 69 has rate \\(\lambda_2\\). 
       
 * The contribution to the expected value will 10% coming from sum(lambda_1_samples_that_have_tau_70) and 90% coming sum(lambda_2_samples_that_have_tau_69)
              

print(lambda_1_samples.mean())
print(lambda_2_samples.mean())

tau_samples = trace['tau']
lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']

N = tau_samples.shape[0]
expected_values = np.zeros(n_counts)
for day in range(0, n_counts):
    # ix is a bool index of all tau samples corresponding to
    # the switchpoint occurring prior to value of 'day'
    ix = day < tau_samples
    # Each posterior sample corresponds to a value for tau.
    # For each day, that value of tau indicates whether we're "before"
    # (in the lambda1 "regime") or
    #  "after" (in the lambda2 "regime") the switchpoint.
    # by taking the posterior sample of lambda1/2 accordingly, we can average
    # over all samples to get an expected value for lambda on that day.
    expected_values[day] = (lambda_1_samples[ix].sum() + lambda_2_samples[~ix].sum()) / N

expected_values

plt.figure(figsize=(12,8))
plt.bar(np.arange(len(total_data)),total_data)
plt.plot(np.arange(n_counts), expected_values, color='g', lw='4')

### GRADED EVALUATION (30 mins)

1. Mean-centering the data helps MCMC Sampling by

    a. Reducing the correlation between the variables (C)
    
    b. Reducing the variance of the variables 
    
    
2. Hierarchical Linear Regression is beneficial when pooling data results in vital group information being lost

    a. True (C)
    
    b. False 
    
    
3. Does this code indicate a 

    a. Hierarchical model
    
    b. Non-hierarchical model (C)
    

```
    α_tmp = pm.Normal('α_tmp', mu=2, sd=5, shape=M)
    β = pm.Normal('β', mu=0, sd=10, shape=M)
    ϵ = pm.HalfCauchy('ϵ', 5)
    ν = pm.Exponential('ν', 1/30)
    
    y_pred = pm.StudentT('y_pred', mu=α_tmp[idx] + β[idx] * x_centered,
                         sd=ϵ, nu=ν, observed=y_m)
   
```

    
  
4. Non-hierarchical Linear Regression with groups of sparse data can result in

    a. very large credible intervals (C)
    
    b. very small credible intervals
    

5. In Hierarchical Linear Regression, priors (not hyperpriors) on variable distributions 

    a. are constant values
    
    b. distributions
    
    
6. Polynomial Regression is useful for

    a. Linear data
    
    b. Non-linear data
    
    
7. Multiple Linear Regression is used when you have

    a. Multiple dependent or target variables
    
    b. Multiple independent or predictor variables (C)
    
    
8. PyMC3 allows you to model multiple predictor variables uing the shape parameter in a distribution without having to create multiple parameters explicitly

    a. True (C)
    
    b. False
    

9. The inverse link function of linear regression is

    a. Logit function
    
    b. Identity function (C)
   
   
10. Multiclass classification uses the

    a. Sigmoid function 
    
    b. Softmax function (C)
    
    
11. A binary classification problem uses the Bernoulli distribution to model the target, the multiclass classification uses

    a. Categorical distribution (C)
    
    b. Poisson distribution

<br>
<br>
<hr style="border:2px solid blue"> </hr>


### MCMC Metrics

#### Paper discussing Bayesian visualization

https://arxiv.org/pdf/1709.01449.pdf

#### Tuning

[Colin Caroll's talk](https://colcarroll.github.io/hmc_tuning_talk/)

When a step size is required, PyMC3 uses the first 500 steps varying the step size to get to an acceptance rate of 23.4%. 

* The acceptance rate is the proportion of proposed values that are not rejected during the sampling process. Refer to Course 2, Metropolis Algorithms for more information. 

* In Metropolis algorithms, the step size is related to the variance of the proposal distribution. See the next section for more information.

These are the default numbers that PyMC3 uses, which can be modified. It was reported in a study that the acceptance rate of 23.4% results in the highest efficiency for Metropolis Hastings. These are empirical results and therefore should be treated as guidelines. According to the SAS Institute, a high acceptance rate (90% or so) usually is a sign that the new samples are being drawn from points close to the existing point and therefore the sampler is not exploring the space much. On the other hand, a low acceptance rate is probably due to inappropriate proposal distribution causing new samples to be rejected. PyMC3 aims to get an acceptance rate between 20% and 50% for Metropolis Hastings, 65% for Hamiltonian Monte Carlo (HMC) and 85% for No U-Turn Sampler (NUTS)

If you have convergence issues as indicated by the visual inspection of the trace, you can try increasing the number of samples used for tuning. It is also worth pointing out that there is more than just step-size adaptation that is happening during this tuning phase. 

`pm.sample(num_samples, n_tune=num_tuning)`

##### Metropolis algorithm
In the Metropolis algorithm the standard deviation of the proposal distribution is a tuning parameter that can be set while initializing the algorithm. 

$x_{t+1} \sim Normal(x_t, stepsize \cdot I)$

The larger this value, the larger the space from where new samples can be drawn. If the acceptance rate is too high, increase this standard deviation. Keep in mind that you run the risk of getting invalid draws if it is too large. 

##### Hamiltonian Monte Carlo (HMC) algorithm

The HMC algorithm is based on the solution of differential equations known as Hamilton's equations. These differential equations depend on the probability distributions we are trying to learn. We navigate these distributions by moving around them in a trajectory using steps that are defined by a position and momentum at that position. Navigating these trajectories can be a very expensive process and the goal is to minimize this computational process.

To get a sense of the intuition behind HMC, it is based on the notion of conservation of energy. When the sampler trajectory is far away from the probability mass center, it has high potential energy but low kinetic energy and when it is closer to the center of the probability mass will have high kinetic energy but low potential energy.

The step size, in HMC, corresponds to the covariance of the momentum distribution that is sampled. Smaller step sizes move slowly in the manifold however larger step sizes can result in integration errors. There is a Metropolis step at the end of the HMC algorithm and and as a result the corresponding target acceptance rates of 65% in PyMC3.

#### Mixing 

Mixing refers to how well the sampler covers the 'support' of the posterior distribution or rather how well it covers the entire distribution. Poor convergence is often a result of poor mixing. This can happen due the choice of 
1. inappropriate proposal distribution for Metropolis 
2. if we have too many correlated variables

The underlying cause for this can be
1. too large a step size to be able to escape a region of high curvature
2. multimodal distributions 


#### Diagnostic statistics

We can compute a metric called Rhat (also called the potential scale reduction factor) that measures the variance between the chains with the variance within the chains. It is calculated as the standard deviation using the samples from all the chains (all samples appended together from each chain) over the  RMS of the within-chain standard deviations of all the chains. Poorly mixed samples will have greater variance in the accumulated samples (numerator) compared to the variance in the individual chains. It was empirically accepted that Rhat values below 1.1 are considered acceptable while those above it are indications of a lack of convergence in the chains. Gelman et al. (2013) introduced a split Rhat that compares the first half with the second half of the samples from each chain to improve upon the regular Rhat. Arviz implements a split Rhat as can be seen from [Arviz Rhat](https://arviz-devs.github.io/arviz/generated/arviz.rhat.html), along with an improved rank-based Rhat
[Improved Rhat](https://arxiv.org/pdf/1903.08008.pdf).

`az.rhat(trace, method='split')`

#### Centered vs. Non-centered Parameterization

[T Wiecki on Reparametrization](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/)

When there is insufficient data in a hierarchical model, the variables being inferred ends up having correlation effects, thereby making it difficult to sample. One obvious solution is to obtain more data, but when this isnt' possible we resort to reparameterization by creating a non-centered model from the centered model. 

This is a centered parameterization for $\beta$ since it is centered around $\mu$

$$\beta \sim Normal(\mu, \sigma)$$

And we try to fit the two parameters for $\mu$ and $\sigma$ directly here.

However, the following is a non-centered parameterization since it is scaled from a unit Normal and shifted

$$\beta_{unit} \sim Normal(0,1)$$

$$\beta = \mu + \sigma \beta_{unit}$$

This works because we are effectively reducing the dependence of \\(\alpha\\) and \\(\beta\\), or making them less correlated, making them easier to sample in a non-centered model. The effect is a joint posterior space that is more rounded.

#### Convergence

Using the example of the centered (cm) and non-centered models (ncm) we can differentiate between models that exhibit good convergence behavior and those that do not.

print("Starting centered model, direct fit")
with pm.Model() as centered_model:
    a = pm.HalfNormal('a', 5)
    b = pm.Normal('b', 0, a, shape=5)
    trace_cm = pm.sample(2000, random_seed=7)
    
print("Starting non-centered model, shifted and scaled model")
with pm.Model() as non_centered_model:
    a = pm.HalfNormal('a', 5)
    b_standard = pm.Normal('b_standard', mu=0, sd=1, shape=5)
    b = pm.Deterministic('b', 0 + b_standard * a)
    trace_ncm = pm.sample(2000, random_seed=7)

print("------------ Centered model ------------")

# The bars indicate the location of the divergences in the sampling process
az.plot_trace(trace_cm, divergences='bottom')
az.summary(trace_cm)

print("------------ Non-centered model ------------")

# The bars indicate the locations of divergences in the sampling process
az.plot_trace(trace_ncm, divergences='top')
az.summary(trace_ncm)

1. The KDE has more agreement for the non-centered model(ncm) compared to the centered model (cm) for the different chains.
2. There are more divergences for cm compared to the ncm as can be seen from the vertical bars in the trace plot. 
3. In general, ncm mixes better than cm - ncm looks like evenly mixed while cm looks patchy in certain regions. 
4. It is possible to see flat lines in the trace, a flat indicates that the same sample value is being used because all new proposed samples are being rejected, in other words the sampler is samping slowly and not getting to a different space in the manifold. The only fix here is to sample for longer periods of time, however we are assuming we can get more unbiased samples if we let it run for longer.

# We plot the densities of both the cm and the ncm models, notice the differences in effective sample sizes for
# the cm (very low)
fig, axs = plt.subplots(1,3)
fig.set_size_inches(18.5, 10.5)
az.plot_forest([trace_cm, trace_ncm], var_names=['a'], 
               kind = 'ridgeplot',
               model_names=['Centered','Non-centered'],
               combined=False, 
               ess=True, 
               r_hat=True, 
               ax=axs[0:3], 
               figsize=(20,20) )
#az.plot_forest(trace_ncm, var_names=['a'],
#               kind='ridgeplot',
#               combined=False, 
#               ess=True, 
#               r_hat=True, 
#               ax=axs[1,0:3])

#### Autocorrelation and effective sample sizes

Ideally, we would like to have zero correlation in the samples that are drawn. Correlated samples violate our condition of independence and can give us biased posterior estimates of our posterior distribution. 
Thinning or pruning refers to the process of dropping every nth sample from a chain. This is to minimize the number of correlated samples that might be drawn, especially if the proposal distribution is narrow. The autocorrelation plot computes the correlation of a sequence with itself but shifted by n, for each n on the x axis the corresponding value of autocorrelation is plotted on the y axis.  

`az.plot_autocorr(trace, var_names=["a", "b"])`

Techniques like Metropolis-Hastings are susceptible to having auto-correlated samples. We plot the autcorrelation here for the cm and the ncm models. The cm models have samples that have a high degree of autocorrelation while the ncm models does not.

fig, axs = plt.subplots(4,2)
fig.set_size_inches(15, 15)
az.plot_autocorr(trace_cm, var_names=['a'], ax=axs[0:4,0])
az.plot_autocorr(trace_ncm, var_names=['a'], ax=axs[0:4,1])

Since a chain with autocorrelation has fewer samples that are independent, we can calculate the number of effective samples called the effective sample size. This is listed during when a summary of the trace is printed out, however it can also be explicitly computed using

`az.efective_n(trace_s)`

PyMC3 will throw a warning if the number of effective samples is less than 200 (200 is heuristically determined to provide a good approximation for the mean of a distribution). Unless you want to sample from the tails of a distribution (rare events), 1000 to 2000 samples should provide a good approximation for a distribution.

#### Monte Carlo error 

The Monte Carlo error is a measure of the error of our sampler which stems from the fact that not all samples that we have drawn are independent. This error is defined by dividing a trace into n blocks. We then compute the mean of these blocks and calculate the error as the standard deviation of these means over the square root of the number of blocks.

$mc_{error} = \sigma(\mu(block_i)) / \sqrt(n)$


#### Divergence

Divergences happen in regions of high curvature or high gradientt in the manifold. When PyMC3 detects a divergence it abandons that chain and as a result the samples that are reported to have been diverging are close to the space of high curvature but not necessarily right on it.

In some cases, PyMC3 can indicate falsely that some samples are divergences, this is due to the heuristics used to identify divergences. Concentration of samples in a region is an indication that these are not divergences. 

We visualize this for the cm and ncm models with pairplots of the variables. You can see how the cm models have difficulty sampling at the edge of the funnel shaped two-dimensional manifold formed by the pairplot. This is a result of the sharp discontinuity where the sampler is having difficulty sampling. 

# Get the divergences
print("Number of divergences %d and percent %lf " % (trace_cm['diverging'].nonzero()[0].shape[0], trace_cm['diverging'].nonzero()[0].shape[0]/ len(trace_cm) * 100))
divergent = trace_cm['diverging']

print("Number of divergences %d and percent %lf " % (trace_ncm['diverging'].nonzero()[0].shape[0], trace_ncm['diverging'].nonzero()[0].shape[0]/ len(trace_ncm) * 100))
divergent = trace_cm['diverging']

az.plot_pair(trace_cm, var_names = ['a', 'b'], divergences=True)
az.plot_pair(trace_ncm, var_names = ['a', 'b'], divergences=True)

az.plot_pair(trace_cm, var_names = ['a', 'b'], coords={'b_dim_0': [0]}, kind='scatter', divergences=True)
az.plot_pair(trace_ncm, var_names = ['a', 'b'], coords={'b_dim_0': [0]}, kind='scatter', divergences=True)

You can also have a parallel coordinates plot of the variables to look at the multidimensional data instead of pairplots. If we notice tight-knit lines around a region, that is an indication of diffculty sampling and hence divergences. This behavior can be observed in cm around 0 while ncm has a sparser cluster of lines around 0. Sparser clusters can be an indication of false positives where divergences are reported. The three ways to avoid the problem of divergences, two of which are changing the parameters passed to PyMC3 and one involves changing how we formulate the problem

1. Increase the tuning samples
2. Increase 'target_accept'

fig, axs = plt.subplots(2,1)
fig.set_size_inches(20,20)
axs[0].set_title('CM model')
axs[1].set_title('NCM model')
az.plot_parallel(trace_cm, var_names=['a','b'], figsize=(20,20), shadend=0.01, colord='tab:blue', textsize=15, ax=axs[0])
az.plot_parallel(trace_ncm, var_names=['a','b'], figsize=(20,20), shadend=0.01, colord='tab:blue', textsize=15,ax=axs[1])

#### A note on why we compute the log of the posterior

In short, this is done to avoid numerical overflow or underflow issues. When dealing with really large or small numbers, it is likely the limited precision of storage types (float, double etc.) can be an issue. In order to avoid this, the log of the probabilities are used instead in the calculations.

### Revisiting the Multiclass Classification Problem

y_s = iris_data.target
x_n = iris_data.columns[:-1]
x_s = iris_data[x_n]
x_s = (x_s - x_s.mean()) / x_s.std()
x_s = x_s.values

import theano as tt
#tt.config.gcc.cxxflags = "-Wno-c++11-narrowing"

with pm.Model() as model_mclass:
    alpha = pm.Normal('alpha', mu=0, sd=5, shape=3)
    beta = pm.Normal('beta', mu=0, sd=5, shape=(4,3))
    μ = pm.Deterministic('μ', alpha + pm.math.dot(x_s, beta))
    θ = tt.tensor.nnet.softmax(μ)
    yl = pm.Categorical('yl', p=θ, observed=y_s)
    trace_s = pm.sample(2000)

data_pred = trace_s['μ'].mean(0)
y_pred = [np.exp(point)/np.sum(np.exp(point), axis=0) for point in data_pred]
az.plot_trace(trace_s, var_names=['alpha'])
f'{np.sum(y_s == np.argmax(y_pred, axis=1)) / len(y_s):.2f}'

#### Diagnostics

All diagnostics in PyMC3 is now in Arviz starting with version 3.9 of PyMC3. Summary is a good place to start

az.summary(trace_s)

#### Rhat

az.rhat(trace_s)

#### Stat Names

Print out the available statistics for your model.

trace_s.stat_names

import seaborn as sns
print("Length of trace_s",len(trace_s.energy), max(trace_s.energy))
print("Depth of the tree used to generate the sample  ",trace_s.depth)
# If tree size is too large, it is an indication of difficulty sampling, due to correlations, sharp posterior space
# or long-tailed posteriors. A solution is to reparameterize the model.
print("Tree size  ",trace_s.tree_size)
print("Energy at the point where the sample was generated ",trace_s.energy)
# This is difference in energy beween the start and the end of the trajectory, should ideally be zero
print("Energy error between start and end of the trajectory ",trace_s.energy_error)
# maximum difference in energy along the whole trajectory of sampling, this can help identify divergences
print("Energy error maximum over the entire trajectory ",trace_s.max_energy_error)
print("Step size ",trace_s.step_size)
print("Best step size determined from tuning ",trace_s.step_size_bar)

#### Trace Energy

Ideally, you want your enerygy and the transition energy to be similar. If your transition energy is too narrow, it could imply that your sampler does not have enough energy to sample the entire posterior space and the sampled results may not appropriately represent the posterior well (biased estimate)

energy_diff = np.diff(trace_s.energy)
sns.distplot(trace_s.energy - trace_s.energy.mean(), label="Energy of trace")
sns.distplot(energy_diff, label="Transition energy")
plt.legend()

# Seaborn uses the interquartile range to draw the box plot whiskers given by Q1 - 1.5IQR, Q3 + 1.5IQR
# The boxplot helps to better visualize the density of outliers
sns.boxplot(trace_s.max_energy_error)

The energy and the energy transition should be as close as possible if the energy transition is smaller or narrower than the marginal energy, it implies that the sampler did not sample the space appropriately and that the results obtained are probably biased.

pm.energyplot(trace_s)

#### Step size

The variation of step size through the sampling process.

sns.lineplot(np.arange(0,len(trace_s.step_size_bar)), trace_s.step_size_bar)

#### Convergence with the Geweke Score

The Geweke score is a z-score that is computed at various segments in the time-series for an inferred parameter. 
A score of less than 2 (less than 2 standard deviations) indicates good convergence. It computes the z-score between each segment and the last 50%, by default, from the sampled chain. The function `pm.geweke` returns an array of (interval start location, z-score)

pm.geweke(trace_s['alpha'])

#### Divergences

# Get the divergences
print("Number of divergences %d and percent %lf " % (trace_s['diverging'].nonzero()[0].shape[0], trace_s['diverging'].nonzero()[0].shape[0]/ len(trace_s) * 100))
divergent = trace_s['diverging']
beta_divergent = trace_s['beta'][divergent]
print("Shape of beta_divergent - Sum of divergences from all chains x shape of variable ", beta_divergent.shape)

import pprint
print("Total number of warnings ",len(trace_s.report._warnings))
pprint.pprint(trace_s.report._warnings[0])

dir(trace_s.report._warnings[0])

print("---------- Message ----------")
pprint.pprint(trace_s.report._warnings[0].message)
print("---------- Kind of warning ----------")
pprint.pprint(trace_s.report._warnings[0].kind)
print("---------- Level ----------")
pprint.pprint(trace_s.report._warnings[0].level)
print("---------- Step ----------")
pprint.pprint(trace_s.report._warnings[0].step)
print("---------- Source ---------- ")
pprint.pprint(trace_s.report._warnings[0].divergence_point_source)
print("---------- Destination ---------- ")
pprint.pprint(trace_s.report._warnings[0].divergence_point_dest)

trace_s.report._warnings[0].divergence_info

for elem in trace_s.report._warnings:
    print(elem.step)

import theano as tt 
import arviz as az 

# If we run into the identifiability problem, we can solve for n-1 variables 
with pm.Model() as model_sf:
    α = pm.Normal('α', mu=0, sd=2, shape=2)
    β = pm.Normal('β', mu=0, sd=2, shape=(4,2))
    α_f = tt.tensor.concatenate([[0] ,α])
    β_f = tt.tensor.concatenate([np.zeros((4,1)) , β], axis=1)
    μ = α_f + pm.math.dot(x_s, β_f)
    θ = tt.tensor.nnet.softmax(μ)
    yl = pm.Categorical('yl', p=θ, observed=y_s)
    trace_sf = pm.sample(1000)
    
az.plot_trace(trace_sf, var_names=['α'])

### Diagnosing MCMC using PyMC3

It is a good idea to inspect the quality of the solutions obtained. It is possible once obtains suboptimal samples resulting in biased estimates or the sampling is slow. There are two broad categories of tests, a visual inspection and a quantitative assessment. Several things that can be done if one suspects issues.

1. More samples, it is possible that there aren't samples to come up with an appropriate posterior
2. Use burn-in, this is removing a certain number of samples from the beginning while PyMC3 is figuring out the step size. This is set to 500 by default. With tuning it is not necessary to explicitly get rid of samples from the beginning.
3. Increase the number of samples used for tuning.
4. Increase the target_accept parameter as 
    `pm.sample(5000, chains=2, target_accept=0.95)`
    
    `pm.sample(5000, chains=2, nuts_kwargs=dict(target_accept=0.95))`
    
    Target_accept is the acceptance probability of the samples. This has the effect of varying the step size in the MCMC process so that we get the desired acceptance probability as indicated by the value to target_accept. It is a good idea to take smaller steps especially during Hamiltonian Monte Carlo so as to explore regions of high curvature better. Smaller step sizes lead to larger acceptance rates and larger step sizes lead to smaller acceptance rates. If the current acceptance rate is smaller than the target acceptance rate, the step size is reduced to increase the current acceptance rates (See the section on tuning).
    
5. Reparameterize the model so that the model while remaining the same, is expressed differently so that is easier to explore the space and find solutions.
6. Modify the data representation - mean centering and standardizing the data are two standard techniques that can be applied here. Note that (5) refers to model transformation while this is specifically modifying the data.

#### Debugging PyMC3

x = np.random.randn(100)

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sigma=1)
    sd = pm.Normal('sd', mu=0, sigma=1)
    
    mu_print = tt.printing.Print('mu')(mu)
    sd_print = tt.printing.Print('sd')(sd)

    obs = pm.Normal('obs', mu=mu_print, sigma=sd_print, observed=x)
    step = pm.Metropolis()
    trace = pm.sample(5, step)
    
trace['mu']

### Arviz Data Representation

From the Arviz page, it states that, apart from NumPy arrays and Python dictionaries, there is support for a few data structures such as xarrays, InferenceData and NetCDF. While NumPy and dictionaries are great for in-memory computations, the other file formats are suitable for persisting computed data and models to disk. InferenceData is a high-level data structure that holds the data in a storage format such as NetCDF.

[Xarray documentation](http://xarray.pydata.org/en/stable/why-xarray.html)

[NetCDF documentation](http://unidata.github.io/netcdf4-python/netCDF4/index.html)

![Structure](https://arviz-devs.github.io/arviz/_images/InferenceDataStructure.png)


#### Load the school data

data = az.load_arviz_data("centered_eight")
data

data.posterior.get('mu')

data = az.load_arviz_data("non_centered_eight")
data

#### Load the S&P500 returns data

sp500 = pd.read_csv(pm.get_data('SP500.csv'), index_col='Date')
sp500

### GRADED EVALUATION (36 min)

1. For Metropolis-Hastings algorithms, an acceptance rate of 23.4 was shown to be ideal

    a. True (C)
    
    b. False
    

2. A high acceptance rate (>90%) is an indication that the sampler is not exploring the space very well

    a. True (C)
    
    b. False
    
    
3. A low acceptance rate is an indication that 

    a. An incorrect proposal distribution is being used (C)
    
    b. The variance of the proposal distribution is too low
    
  
4. When using the NUTS algorithm, PyMC3 aims to get an acceptance rate of 

    a. 75%
    
    b. 85% (C)
    
    
5. If you have convergence issues, it is better to

    a. Try increasing the total number of samples drawn
    
    b. Try increasing the number of tuning samples (C)
    
    
6. A step size that is too large can result in 

    a. Large sample values
    
    b. Invalid sample values (C)
    
    
7. Large step sizes in Hamiltonian Monte Carlo can result in 

    a. Integration errors (C)
    
    b. Out-of-bounds errors
    
    
8. Mixing in MCMC refers to

    a. How well the sampling covers the entire distribution space (C)
    
    b. The similarity of the sample values
    
    
9. Rhat, used to measure mixing measures 

    a. the variance between the chains
    
    b. the ratio of the variance between the chains to the variance within the chains (C)
    
    
10. Rhat values below 1.1 indicate convergence while those above do not

    a. True (C)
    
    b. False
    
    
11. Thinning or pruning refers to dropping every n'th sample to avoid correlated samples

    a. True (C)
    
    b. False
    
    
12. Divergences happen in regions of high curvature or sharp gradients in the sampling manifold

    a. True (C)
    
    b. False

