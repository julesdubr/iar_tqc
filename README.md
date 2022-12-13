# TQC Analysis

- Original paper :
    - (v1) <https://arxiv.org/pdf/2005.04269.pdf>
    - (v2) <http://proceedings.mlr.press/v119/kuznetsov20a/kuznetsov20a.pdf>
- Original Implementation:
    - <https://github.com/bayesgroup/tqc>
    - <https://github.com/bayesgroup/tqc_pytorch>


# Plan d'action

## 1. Learning curves TQC vs SAC (_Figure 5_)

- [ ] Hopper-V3
    - 1M frames
    - 5 seeds : 42, 2, 32, 20, 17
    - TQC : ~ 10h / run
    - SAC : ~ 5h / run
    - total : ~ 75h

- [ ] Humanoid-V3
    - 3M frames
    - ...
    - total : ~ 225h


## 2. Robust average of bias/variance of Q-function approx (_Figure 4_)

- [ ] Implémenter "Toy experiment" (Appendix C + Section 4.1)
- [ ] Reproduire la courbe


## 3. Overestimation in MuJoCo (_Figure 6_)

- Dépendance de la performance et de la distribution l'estimation de l'erreur selon le nombre d'atomes $d$ enlevé.


# Préparation / Questions

__Préparation__

- [ ] Comprendre section 4.1 sur l'implémentation du MDP
- [ ] Réinitialiser les sessions PPTI
- [ ] Setup ordi de Julien


__Questions__

- Plan OK ?
    - Résultats à présenter suffisants ?
    - Autres tests à effectuer / courbes à reproduire (vis à vis du papier) ?
    - Sortir des résultats du papier (tester d'autres choses) ?
- Setup de l'environnement sur les machines de la PPTI (sudo...) ?
    - Sinon quelles sont les autres solutions pour réduire les temps de calcul ?
- Idées sur l'implémentation du single MDP ?