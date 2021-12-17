# Deep RL Knowledge Base 🦾 

## A. Value-Based 📈

### 1. DQN 👾

#### High-Level Description

Learn a state-action value function parameterized by a neural network. Optimal policies can be recovered by acting greedily _w.r.t._ the state-action values.

#### Algorithm Details

$\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{s, a, s^\prime} \left[\left(r + \gamma \max_{a^\prime}Q(s^\prime, a^\prime; \theta^\prime) - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta)\right]$, where $\theta^\prime$ represents parameters of the target network.

#### Implementation Details

- Trick 1: Experience replay. Alleviate the problem of distributional shift to some extent.
- Trick 2: Target network. Stabilize the regression target.
- Trick 3: Frame stack. Enable the model to infer information that requires history, e.g., velocity.

### 2. Double DQN ‼️

#### High-Level Description

Reduce overestimations of Q-values by decomposing the max operation in the target into action selection and action evaluation.

#### Algorithm Details

Use online network with parameters $\theta$ to propose actions but use the target network with parameters $\theta^\prime$ to evaluate Q-values.
$$
\begin{align}
a^\prime &= \arg\max_{a^\prime} Q(s^\prime, a^\prime; \theta) \\
target &= r + \gamma Q(s^\prime, a^\prime; \theta^\prime)
\end{align}
$$

### 3. Dueling DQN

#### High-Level Description

Enhance generalization by separately learning a state-dependent advantage function and a state value function.

#### Algorithm Details

$Q(s, a; \theta, \phi) = V(s; \theta) + \left(A(s, a; \phi) - \frac{1}{\vert \mathcal{A} \vert} \sum_{a^\prime} A(s, a^\prime; \phi) \right)$

Reasons for subtracting the averaged advantages

1. The advantage should actually be learned as $A(s, a; \phi) - \max_{a^\prime} A(s, a^\prime; \phi)$ because for optimal policy, state value should be equal to state-action value, i.e., the advantage of optimal action should be evaluated to zero. Subtracting the max value of advantage satisfies this property.
2. Replace the max advantage being subtracted with the average advantage increases the stability of optimization. Because in this case the advantage function only need to change as fast as the averaged advantage changes, instead of having to compensate any change to the optimal action's advantage.

## B. Policy Gradient-Based 𝚫

### 1. Policy Gradient (REINFORCE)

Definition: The gradient of policy performance $\nabla_\theta J(\pi_\theta)$, where $J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$.

Derivation
$$
\begin{align}
\nabla_\theta J(\pi_\theta) &= \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] \\
&= \nabla_\theta \int P(\tau) R(\tau) d\tau \\
&= \int \nabla_\theta P(\tau)R(\tau)d\tau \\
&= \int P(\tau) \nabla_\theta\log P(\tau) R(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log P(\tau) R(\tau)]\\
&= \mathbb{E}_{\tau \sim \pi_\theta} [\sum_t^T \nabla_\theta \log \pi_\theta(a_t \vert s_t) R(\tau)]
\end{align}
$$

### 2. Policy Gradient with reward to go

Replace the trajectory return with the cumulative reward evaluated starting from tilmestep $t$: $\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [\sum_t^T \nabla_\theta \log \pi_\theta (a_t \vert s_t) \sum_{t^\prime = t}^T R(s_{t^\prime}, a_{t^\prime}, s_{t^\prime + 1})] $

### 3. Reward-to-go with baseline

Adding or subtracting a baseline does not affect the policy gradient under expectation as long as the baseline is *state-dependent*. 

$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [\sum_t^T \nabla_\theta \log \pi_\theta (a_t \vert s_t) \sum_{t^\prime = t}^T R(s_{t^\prime}, a_{t^\prime}, s_{t^\prime + 1}) - b(s_{s^\prime})] $

### 4. Vanilla PG

Key idea: Push up the probabilities of actions that lead to higher return and push down the probabilities of actions that lead to lower return.

Use advantage as weights in the policy gradient.

Some issue: may get stuck at the local optima because the exploration is only dependent on the randomness of the policy. As training goes on, the policy tends to be less random and to exploit high returns.

### 5. TRPO

Key idea: A constrained version of vanilla PG that the updated policy should not be too far away from the old policy. The policy different is measured by KL divergence. TRPO will take the largest possible step to improve the performance.

### 6. PPO(-clip)

Key idea: Same motivation as TRPO, but satisfy the policy update constraint in an easier way: clip the advantage function.

Objective function
$$
L(s, a, \theta, \theta^\prime) = \min \left(\frac{\pi_\theta (a \vert s) }{\pi_{\theta^\prime}(a \vert s)}A^{\pi_\theta^\prime} (s, a), g(\epsilon, A^{\pi_{\theta^\prime}} (s, a))  \right),
$$
where $g(\epsilon, A) = (1 + \epsilon) A$ if $A > 0$ else $(1 - \epsilon )A$.

Intuition:

1. When advantage is positive, increasing $\pi_\theta(a \vert s)$ can improve the performance. But we cannot increase $\pi_\theta(a \vert s)$ too much, the maximum value we can increase to is $(1 + \epsilon) \pi_\theta ( a \vert s)$.
2. When advantage is negative, decreasing $\pi_\theta (a \vert s)$ can improve the performance. But we cannot decrease $\pi_\theta(a\vert s)$ too much, the minimum value we can decrease to is $(1 - \epsilon)\pi_\theta(a \vert s)$.

### 7. A3C

Key idea: Asynchronously execute multiple agents in parallel on multiple instances of environment.

Benefits: 1. Stabilize training due to parallelism 2. Speed up in terms of wall-clock time using less computation.

Key equation: Use n-step look-ahead to calculate the advantage in policy gradients. Also use policy entropy as a heuristic regularization to encourage exploration. 

