---
layout: post
title: Reinforcement Learning Basic
date: 2021-02-22 16:24 +0800
toc: true
---


## Terminology & Notation


* \\( s_{t} \\) - state
* \\( o_{t} \\) - observation
* \\( a_{t} \\) - action
* \\( \pi_{\theta} \left( a_t \mid o_t \right) \\) - policy
* \\( \pi_{\theta} \left( a_t \mid s_t \right) \\) - policy (fully observerd)
* \\( r \left( s_t, a_t \right) \\) - reward function
* \\( \hat{Q}_{t} \\) - reward to go


## The Goal of Reinforcement Learning

Finite horizon case:

$$

\underbrace{p_{\theta} \left( s_{1}, a_{1}, \dots, s_{T}, a_{T} \right)}_{p_{\theta} \left( \tau \right) }
=p \left( s_{1} \right) \prod_{t=1}^{T} \underbrace{\pi_{\theta} \left( a_{t} \mid s_{t} \right)  p \left( s_{t+1} \mid s_{t}, a_{t} \right) }_{\text{Markov chain on} \left( s, a \right) }

$$

$$

\begin{aligned}
\theta^{*} &= \mathop{\arg\max}_{\theta} E_{ \tau \sim p_{\theta} \left( \tau \right) } \left[ \sum_{t}{r \left( s, a \right) } \right] \\
           &= \mathop{\arg\max}_{\theta} \sum_{t=1}^{T}{
             E_{ \left( s_{t}, a_{t} \right) \sim p_{\theta} \left( s_{t}, a_{t} \right) } \left[  r \left( s_{t}, a_{t} \right) \right]
           } && \text{here  $p_{\theta} \left( s_{t}, a_{t} \right)$ is state-action marginal} \\
\end{aligned}

$$

Infinite horizon case:

$$
\theta^{*} = \mathop{\arg\max}_{\theta} E_{ \left( s, a \right) \sim p_{\theta} \left( s, a \right) } \left[ r \left( s, a \right) \right]
$$


## Policy Gradient

### Evaluate the Objective

$$
J \left( \theta \right) = E_{\tau \sim p_{\theta} \left( \tau \right) } \left[ \sum_{t}{r \left( s_t, a_t \right) } \right]
\approx \frac{1}{N} \sum_{i}{\sum_{t} r \left( s_{i, t}, a_{i, t} \right) }
$$


### Direct Policy Differentiation

$$
J \left( \theta \right) = E_{\tau \sim p_{\theta} \left( \tau \right) } [ \underbrace{r \left( \tau \right) }_{ \sum_{t=1}^{T}{r \left( s_t, a_t \right) } } ]
= \int{p_{\theta}\left( \tau \right) r \left( \tau \right) d\tau}

$$

$$

\begin{aligned}

\nabla_{\theta} J \left( \theta \right)
&= \nabla_{\theta} \int {p_{\theta}\left( \tau \right) r \left( \tau \right) d\tau} \\
&= \int \nabla_{\theta} {p_{\theta}\left( \tau \right) r \left( \tau \right) d\tau} \\
&= \int p_{\theta} \left( \tau \right) \nabla_{\theta} \log {p_{\theta}} \left( \tau \right) r \left( \tau \right) d\tau \\
&= E_{\tau \sim p_{\theta} \left( \tau \right) } \left[ \nabla_{\theta} \log {p_{\theta}} \left( \tau \right) r \left( \tau \right) \right] \\
&= E_{\tau \sim p_{\theta} \left( \tau \right) } \left[ \nabla_{\theta} \left[
    \log p \left( s_1 \right) + \sum_{t=1}^{T}{\log \pi_{\theta} \left( a_t \mid s_t \right) + \log p \left( s_{t+1} \mid s_t, a_t \right)}
\right] r \left( \tau \right) \right] \\
&= E_{\tau \sim p_{\theta} \left( \tau \right) } \left[
\left(
    \sum_{t=1}^{T}{\nabla_{\theta} \log \pi_{\theta} \left( a_t \mid s_t \right) }
\right)
\left( \sum_{t=1}^{T}{r \left( s_t, a_t \right)} \right)
\right] \\

\end{aligned}
$$

So we can derive the REINFORCE algorithm:

{% include pseudocode.html id="1" code="
\begin{algorithm}
\caption{REINFORCE}
\begin{algorithmic}
\FOR{episodes $ \{ \tau^{i} \} \sim \pi_{\theta} \left( a_{t} \mid s_{t} \right)$}
    \STATE $ \nabla_\theta J \left( \theta \right) \approx \sum_{i}{\left( \sum_{t}{\nabla_\theta \log \pi_\theta(a_{t}^{i} \mid s_{t}^{i})} \right) \left( \sum_{t}{ r \left( s_{t}^{i}, a_{t}^{i} \right) } \right) } $
    \STATE $ \theta \leftarrow \theta + \alpha \nabla_\theta J \left( \theta \right) $
\ENDFOR
\end{algorithmic}
\end{algorithm}
" %}

### Partial Observability

Similar to the derivation above we can replace \\( s_{t} \\) to \\( o_{t} \\):

$$
\nabla_{\theta} J \left( \theta \right)
\approx \frac{1}{N} \sum_{i=1}^{N}
\left(
    \sum_{t=1}^{T}{\nabla_{\theta} \log \pi_{\theta} \left( a_t \mid o_t \right) }
\right)
\left( \sum_{t=1}^{T}{r \left( s_t, a_t \right)} \right)
$$

The Markov property is not actually used.


### Causality

Policy at time \\( t \\) cannot affect reward before.

$$
\begin{aligned}
\nabla_{\theta} J \left( \theta \right)
&\approx \frac{1}{N} \sum_{i=1}^{N}
    \sum_{t=1}^{T}{\nabla_{\theta} \log \pi_{\theta} \left( a_t \mid s_t \right) \left( \sum_{t=1}^{T}{r \left( s_t, a_t \right)} \right) } \\
&\approx \frac{1}{N} \sum_{i=1}^{N}
    \sum_{t=1}^{T}{\nabla_{\theta} \log \pi_{\theta} \left( a_t \mid s_t \right) \underbrace{ \left( \sum_{t^{\prime}=t}^{T}{r \left( s_{t^{\prime}}, a_{t^{\prime}} \right)} \right) }_{\text{reward to go}} } \\
&= \frac{1}{N} \sum_{i=1}^{N}
    \sum_{t=1}^{T}{\nabla_{\theta} \log \pi_{\theta} \left( a_t \mid s_t \right) \hat{Q}_{i, t} } \\
\end{aligned}
$$


### Baseline

Actually we can substract a baseline \\( b \\) from reward of trajectory \\( \tau \\):

$$
\nabla_{\theta} J \left( \theta \right)
\approx \frac{1}{N} \sum_{i=1}^{N}{ \nabla_{\theta} \log p_{\theta} \left( \tau \right) \left( r \left( \tau \right) - b \right) }
$$

Because:

$$
-\int p_{\theta} \left( \tau \right) \nabla_{\theta} \log p_{\theta} \left( \tau \right) b d\tau
= -\int \nabla_{\theta} p_{\theta} \left( \tau \right) b d\tau
= -b \int \nabla_{\theta} p_{\theta} \left( \tau \right) d\tau
= -b \nabla_{\theta} \int p_{\theta} \left( \tau \right) d\tau
= -b \nabla_{\theta} 1
= 0
$$

So baseline is unbiased. A typical choice of baseline is:

$$
b = \frac{1}{N} \sum_{i=1}^{N}{r \left( \tau \right)}
$$

### Variance

$$
\mathrm{Var} \left( x \right) = E \left[ x^2 \right] - {E \left[ x \right]}^2
$$

$$
\nabla_{\theta} J \left( \theta \right)
= E_{\tau \sim p_{\theta} \left( \tau \right)}
\left[
{ \nabla_{\theta} \log p_{\theta} \left( \tau \right) \left( r \left( \tau \right) - b \right) }
\right]
$$

So the variance is:

$$
\mathrm{Var}

= E_{\tau \sim p_{\theta} \left( \tau \right)} \left[ \Big( \nabla_{\theta} \log p_{\theta} \left( \tau \right) \left( r \left( \tau \right) - b \right) \Big)^2 \right]
    - {E_{\tau \sim p_{\theta} \left( \tau \right)} \left[ \underbrace{\nabla_{\theta} \log p_{\theta} \left( \tau \right) \left( r \left( \tau \right) - b \right)}_{
    \substack{
        \text{this bit is just $E_{\tau \sim p_{\theta} \left( \tau \right)} \left[ \nabla_{\theta} \log p_{\theta} \left( \tau \right) r \left( \tau \right)  \right] $}
        \\ \text{because baseline is unbiased in expection}
    }
} \right]}^2
$$

We can derive the derivative for \\(b\\):

$$
\begin{aligned}
\frac{d\mathrm{Var}}{db}
&= \frac{d}{db} E \left[ g \left( \tau \right)^2 \left( r \left( \tau \right) - b \right)^2 \right] - {E \left[ g \left( \tau \right) \left( r \left( \tau \right) - b \right) \right]}^2 \\
&= \frac{d}{db} E \left[ g \left( \tau \right)^2 r \left( \tau \right)^2 - 2 g \left( \tau \right)^2 r \left( \tau \right) b + g \left( \tau \right)^2 b^2 \right]
    - {E \left[ g \left( \tau \right) r \left( \tau \right) \right]}^2 \\
&= - 2 E \left[ g \left( \tau \right)^2 r \left( \tau \right) \right] + 2b E \left[ g \left( \tau \right)^2 \right]
\end{aligned}
$$

Then, we can get the optimal baseline, here \\( g \left( \tau \right) = \nabla_{\theta} \l{\theta} \left( \tau \right) \\):

$$
b = \frac{ E \left[ g \left( \tau \right)^2 r \left( \tau \right) \right] }{ E \left[ g \left( \tau \right)^2 \right] }
$$

