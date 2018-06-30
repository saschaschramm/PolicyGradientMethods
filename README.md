# PolicyGradientMethods

## Introduction

Reinforcement learning methods that learn a parameterized policy. 
These methods learn by approximating the gradient of a performance measure with respect to its policy parameters.

## Reinforce
Example with T = 3:
```
S0, A0 -> R1
S1, A1 -> R2
S2, A2 -> R3

t = 0: G0 = discount_rate^0 * R1 + discount_rate^1 * R2 + discount_rate^2 * R3
t = 1: G1 = discount_rate^0 * R2 + discount_rate^1 * R3
t = 2: G2 = discount_rate^0 * R3

t = 0: discount_rate^0 * G0
t = 1: discount_rate^1 * G1
t = 2: discount_rate^2 * G2
```

