# TQC Analysis

- Original paper :
    - (v1) <https://arxiv.org/pdf/2005.04269.pdf>
    - (v2) <http://proceedings.mlr.press/v119/kuznetsov20a/kuznetsov20a.pdf>
- Original Implementation:
    - <https://github.com/bayesgroup/tqc>
    - <https://github.com/bayesgroup/tqc_pytorch>


# Plan d'action

The bias of overestimation is one of the main obstacles to the accuracy of off-policy learning methods. Indeed, approximation errors are retro-propagated and accumulate during the learning process.

In this context, Truncated Quantile Critics (_TQC_) is a new (2020) way to attenuate the effects of overestimation in a continuous action space. It is based on three ideas: the distributive representation of the critic, the truncation of critic predictions, and the mixture of critics.

## Soft Actor Critic

SAC is the algorithm on which TQC is implemented. It is an off-policy actor-critic algorithm that promotes exploration based on entropy to measure the level of randomness of values.

If the entropy is high (high disorder), it means we have not explored enough, so we increase the "value" of the state to promote exploration. The temperature (impact) of entropy is regulated by a coefficient $\alpha$ that adjusts dynamically during learning. It decreases when the estimated entropy of the policy is greater than the targeted entropy $\mathcal{H}_T = - \dim \mathcal{A}$ and increases otherwise. By increasing the reward based on entropy, the algorithm encourages the stochasticity of the policy.

## Distributional representation

A distribution of estimates is more stable than a single estimate. Therefore, Distributional RL focuses on approximating the Q-value as a random variable $Z^\pi(s,a)$ rather than an expectation estimate, $Q^\pi(s,a) := \mathbb{E} [Z^\pi(s,a)]$.

The approximation of the distribution $Z^\pi(s,a)$ is performed by a mixture of atoms, based on Dirac distributions according to a parametric model $\theta_\psi$. The optimization of the parameters $\psi$  is carried out by quantile regression, allowing for the approximation of the quantiles of $Z(s,a)$.

This method allows for learning the intrinsic randomness of the environment and policy (random uncertainty).

## Truncated Quantile Critics

TQC differs from SAC in its method of evaluating the critic using the distributional representation and the number of critics. To improve Q-value estimation, TQC forms a mixture of distributions (of $M$ atoms each) predicted by $N$ critics. This gives an estimated distribution of $NM$ atoms.

To control under- or overestimation, TQC then truncates the $dN$, $d \in \{1..M\}$, estimated Q-values (atoms) with the largest value and estimates a mean Q-value from the remaining atoms.

This method does not impose any restrictions on the number of approximators required, allowing for the separation of overestimation control (truncation) from the assembly of distributions (mixture).

## Results

It is observed that using the minimum or mean over N Q-networks in SAC leads to overestimation, whereas using TQC's truncation method leads to improved performance and reduced overestimation. In addition, TQC allows for more efficient use of resources and faster convergence compared to SAC with multiple Q-networks.