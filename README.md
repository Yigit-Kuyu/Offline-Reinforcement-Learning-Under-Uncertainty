### INFO
This repository implements various offline RL algorithms, tested on D4RL benchmark datasets. Results include text files, figures, and trained networks.

*Codes are standalone and GPU-compatible.*

### [D4RL](https://github.com/Farama-Foundation/D4RL)
Benchmark datasets for offline RL.

### Offline RL Algorithms

#### [BEAR](https://arxiv.org/abs/1906.00949)
- Prevents out-of-distribution actions
- Constrains learned policy to stay close to behavior policy
- Uses distributional regularization to minimize MMD

#### [DiffCPS](https://arxiv.org/abs/2310.05333)
- Integrates diffusion models into offline RL
- Addresses limited expressivity of Gaussian-based policies
- Uses primal-dual approach for constrained policy search

#### [TD3_BC](https://arxiv.org/abs/2106.06860)
- Modifies TD3 with behavior cloning loss
- Guides policy to stay close to observed actions
- Uses policy gradient approach for optimization

#### [BCQ](https://arxiv.org/abs/1812.02900)
- Tackles distributional shift in offline RL
- Uses VAE to model behavior policy
- Selects actions likely under behavior policy



### Results
- Gradient steps: 1e4, Batch size: 100, Environment: halfcheetah-random-v2
- Further comparison with different seeds needed

| Algorithm | Average Return  |  D4RL Score |
|-----------|-----------------|-------------|
| BEAR      |   2811.5        | 25.9        |
| DiffCPS   |   1618.5        | 14.6        |
| TD3_BC    |   1100.9        | 11.1        |
| BCQ       |   0.5           | 2.3         |



![Figure](https://github.com/Yigit-Kuyu/Offline-Reinforcement-Learning-Under-Uncertainty/blob/main/Comparative_testing_curve_normalized.png)






    


