[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"
[image3]: imgs/scores.png "scores"
[image4]: imgs/tennis.gif "tennis"
## The environment


The environment to solved is the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image4]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Training

For this environment, the Distributed Distributional Deep Deterministic Policy Gradient algorithm [D4PG](https://openreview.net/pdf?id=SyZipzbCb) is used, this algorithm is at its core an off-policy actor-critic method. This means that we consider a parametrized actor-function (Implemented as a DNN in the `model.py` file) which specifies the current policy by deterministically mapping states to a specific action. At the same  time, a critic that estimates the action-value function (Q(s,a)) is learned and is used to estimate the actor's loss function as a baseline for the expected Reward. 

Same as in [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), this algorithm also uses:


1. Memory Replay: Several steps are stored on a memory buffer and then randomly sampled for training as defined in the `ReplayBuffer` class on the file  `model/dqn_agent.py`, hence coping with state-action correlation.

2. Fixed Q-Targets: This is done by means of a second network (same architecture as the DQN) that is used as a target network. However instead of fixing and updating the target network every n-steps, a soft update is performed.

3. Aditionally, Batch Normalization is used

For more details, check the [D4PG paper](https://openreview.net/pdf?id=SyZipzbCb).

Scores:

![Scores][image3]

#### Hyperparameters

The main parameters defined for the D4PG algorithm used for this environment are:
```python
lr_actor = 1e-3   #Learning Rate used for actor's Adam Optimizer
lr_critic = 1e-3  #Learning Rate used for critic's Adam Optimizer
gamma = 0.99      #Disscount factor
tau = 1e-3        #Soft update factor
batch_size =512   #Size of batches used for training
buffer_size = int(1e5)   #Size of the replay buffer
update_every = 20   #The agent will be trained every n steps
num_updates = 10    #Number of batches on which agent will be trained                      #every time
```

## Future work
As future work, MADDPG with a centralized critic could be used. Additionally other techniques such as infering policies for other agents instead of using our global knowledge of the environment or using agents with Policy Ensembles should give interesting results.

Furthermore, since a ReplayBuffer is being used (off-policy), techniques such as prioritized learning should also improve the performance and convergence time of the agent