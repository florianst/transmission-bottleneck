# Bottleneck Size Estimation
Result of the 2019 Interdisciplinary Bioscience Hackathon (Oxford)

We implemented a simulation of pathogen replication dynamics upon infection based on Leonard et al., *Transmission Bottleneck Size Estimation from Viral Sequencing Data*, Journal of Virology **91**, 14 (2017). We obtain variant sites for two different viruses (H1N1, H3N2) and, using binomial (binomial_AND_beta_binomial.nb) and beta-binomial (betabinomial.py) sampling methods, try to infer the most likely bottleneck size as the maximum likelihood value of the resulting distribution. 

We check the sanity of our algorithms with computationally generated viral data for a bottleneck size of N_b=50 (generating_simulated_bottlenecks.nb,  	donorFreq.txt, recipientFreq.txt) and run a population dynamics simulation for three viral populations (ca_all.py).

## ToDo
tidy repository!


