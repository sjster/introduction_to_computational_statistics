# Bayesian vs Frequentist statistics

Likelihood is a type of probability except that this describes the probability of the data has already been observed. Probability, in general, refers to the possibility of occurence of future events. This is the basis on frequentist statistics, our probabilities are only based on what we have observed. In Bayesian statistics, we also use our prior understanding of the environment or the system that generated the observations to infer the possibility of future events. An Ben Lambert says in his book 'A Students Guide to Bayesian Statistics', 'for Bayesians it is unncessary for events to be repeatable in order to define a probability. They are merely abstractions to help express our uncertainty'.

Bayesians - Data is fixed and in light of new data we update our beliefs. The data is fixed and the beliefs are updated. The belief represented by some parameter is a random variable that we update. They are comfortable with the idea that our knowledge of these parameters evolve over time. In the case where there is a true parameter value that can be estimated, Bayes theorem allows one to specify an uncertainty over this parameter.

Frequentist - The data is result of sampling from a random process. They see the data as varying and the parameter of this random process that generates the data as being fixed. They view this parameter as being the average of a infinite number of experiments. This approach is particlary problematic for events that cannot be repeated.

His example of the US elections helps to illustrate this subtle difference.

Bayesian - The probability of the Democrats winning the election is 0.75

Frequentist - Since there is only sample that can be obtained for a particular election, this is not a repeatable exercise and we cannot sample from a population of outcomes for this election.

For e.g. in a coin flip that is repeated 5 times where 4 heads show up, the probability of heads for a frequentist would be 4/5. If O represents the outcome and H represents the outcome of heads then the likelihood can be written as

$Likelihood = P(O=H/\theta)$ = 0.8 ( (Frequentists use this as the probability for their hypothesis )

where $\theta$ represents the parameter of our coins probability density (Binomial distribution)

However a Bayesian approach would consider the belief that the coin is unbiased and there has a prior probability ($P(\theta)$) of 0.5 of a heads showing up and have the following probability 

$Posterior = Likelihood \times Prior = P(\theta | O = H) \propto P( O = H | \theta) \times P(\theta)$ (Bayesians use this )

$Posterior \propto 0.8 * 0.5$

Suppose that you are measuring the heights of citizens in your state and there is a hypothesis that ther heights are normally distributed with mean  5'10" and a standard deviation of 4". What is the probability of a person having a height of 6'8"? Frequentists would have to compute the probability of this observation happening as $P(h >= 6.8 | N(\mu, \sigma) )$ and use a threshold to determine if our original hypothesis was valid. In Bayesian statistics, the posterior is a probability distribution of the parameters of our hypothesis given our observation of seeing a sample with height 6'8" $P(N(\mu, \sigma) | h = 6'8")$. However, in this case we use a subjective prior to compute this posterior. We trade an arbitrary threshold (usually 1% or 5% depending on the problem being solved) in frequentist statistics for a subjective prior in bayesian statistics. One could argue that both of these are different ways to incorporate domain-knowledge and subjective expertise into the decision making process. Bayesians however do not need to accept or reject a hypothesis since they have the full distribution available as a result of the posterior and can therefore quantify the uncertainty with the hypothesis.

