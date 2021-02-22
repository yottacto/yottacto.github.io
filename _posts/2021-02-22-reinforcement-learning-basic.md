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


## The Goal of Reinforcement Learning

Finite horizon case:

$$

\underbrace{p_{\theta} \left( s_{1}, a_{1}, \dots, s_{T}, a_{T} \right)}_{p_{\theta} \left( \tau \right) }
=p \left( s_{1} \right) \prod_{t=1}^{T} \underbrace{\pi_{\theta} \left( a_{t} \mid s_{t} \right)  p \left( s_{t+1} \mid s_{t}, a_{t} \right) }_{\text{Markov chain on} \left( s, a \right) }

\\

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

\\

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

