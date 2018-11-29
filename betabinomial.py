import numpy as np
import pandas as pd
from scipy.special import gamma, gammaln
from scipy.stats import beta, binom
import matplotlib.pyplot as plt

# calculate the log-likelihood of a certain bottleneck size Nb for given variant frequencies in donor and recipient,
# based on the combination of a beta and a binomial distribution

# read results from Leonard et al., J. Virology 91 (2017)
df1 = pd.read_csv("data2016_h1n1.csv", sep='\t', header=(0), dtype=int, usecols=[0,1,2,3,6,7,8,9,10,11]) # exclude allele letters
df2 = pd.read_csv("data2016_h3n2.csv", sep='\t', header=(0), dtype=int, usecols=[0,1,2,3,6,7,8,9,10,11])

# calculate variant frequencies in donor and recipient
df1['variantfreq_R'] = df1["Variant reads in recipient"]/df1["Total number of reads in recipient"]
df1['variantfreq_D'] = df1["Variant reads in donor"]/df1["Total number of reads in donor"]

df2['variantfreq_R'] = df2["Variant reads in recipient"]/df2["Total number of reads in recipient"]
df2['variantfreq_D'] = df2["Variant reads in donor"]/df2["Total number of reads in donor"]

# clean data: ignore data points where variantfreq_R > 3% and variantfreq_D < 3% - assume these to be mutations which appeared after transmission
cutoff = 3.0/100
for i in range(df1['variantfreq_R'].shape[0]):
    if (df1['variantfreq_R'][0] > cutoff and df1['variantfreq_D'][0] < cutoff): df1 = df1.drop(df1.index[i])
for i in range(df2['variantfreq_R'].shape[0]):
    if (df2['variantfreq_R'][0] > cutoff and df2['variantfreq_D'][0] < cutoff): df2 = df2.drop(df2.index[i])

def likelihood_approx(df, Nb):
    return sum(np.nan_to_num(beta.pdf(df['variantfreq_R'], k, Nb-k)) * binom.pmf(k, Nb, df['variantfreq_D']) for k in range(Nb))

def log_likelihood_approx(df, Nb): # multiply along sites
    return np.sum(np.log(likelihood_approx(df, Nb)))

points = np.arange(5,250,5)
likelihoods1 = [log_likelihood_approx(df1, Nb) for Nb in points]
likelihoods2 = [log_likelihood_approx(df2, Nb) for Nb in points]

likelihoods1 = likelihoods1/abs(max(likelihoods1))
likelihoods2 = likelihoods2/abs(max(likelihoods2))

plt.plot(points, likelihoods1, label='H1N1')
plt.plot(points, likelihoods2, label='H3N2')
plt.xlabel('bottleneck size $N_b$')
plt.ylabel('normalised log-likelihood $L(N_b)$')
plt.legend()

plt.savefig('approximate_bottleneck_inference.png')
plt.show()