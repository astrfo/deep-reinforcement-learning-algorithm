# deep-reinforcement-learning-study

深層強化学習アルゴリズムの実装

## 実装済み(中)アルゴリズム

- DQN: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236.pdf)
- DoubleDQN: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- PrioritizedExperienceReplay: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- DuelingNetwork: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- CategoricalDQN(C51): [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
- NoisyNetwork: [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
- SimplePolicyGradient: [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696)
- REINFORCE: [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696)
- Actor-Critic: [Witten(1977); Barto, Sutton, Anderson(1983); Sutton(1984)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- RandomNetworkDistillation(RND): [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
- GORILA: [Massively Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1507.04296)

## 使用方法

```bash
pip install -r requirements.txt
python main.py
```

## アルゴリズム選択

`simulator.py`の`self.policy`に使用したいアルゴリズムのクラスを定義．  
ただし，`PolicyGradient`と`REINFORCE`は`PGSimulation`クラスを使用し，`ActorCritic`は`ACSimulation`クラスを使用してください．(`main.py`も適宜変更してください)

```python
def __init__(self, sim, epi, env):
    ...
    self.policy = DQN()
```

## 注意事項

ハイパーパラメータやGym環境，指標，実験設定の保存等，そこら辺はかなりいい加減に書いているので注意が必要．  
実装の練習にもなると思うので余力がある人は自分でカスタマイズしてみてください．  
