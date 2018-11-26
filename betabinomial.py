import numpy as np
import pandas as pd
from scipy.special import gammaln

# calculate the log-likelihood of a certain bottleneck size Nb for given variant frequencies in donor and recipient,
# based on the combination of a beta and a binomial distribution
