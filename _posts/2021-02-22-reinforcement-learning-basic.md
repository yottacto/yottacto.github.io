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

{% include pseudocode.html id="1" code="
\begin{algorithm}
\caption{Quicksort}
\begin{algorithmic}
\PROCEDURE{Quicksort}{$A, p, r$}
    \IF{$p < r$}
        \STATE $q = $ \CALL{Partition}{$A, p, r$}
        \STATE \CALL{Quicksort}{$A, p, q - 1$}
        \STATE \CALL{Quicksort}{$A, q + 1, r$}
    \ENDIF
\ENDPROCEDURE
\PROCEDURE{Partition}{$A, p, r$}
    \STATE $x = A[r]$
    \STATE $i = p - 1$
    \FOR{$j = p$ \TO $r - 1$}
        \IF{$A[j] < x$}
            \STATE $i = i + 1$
            \STATE exchange
            $A[i]$ with     $A[j]$
        \ENDIF
        \STATE exchange $A[i]$ with $A[r]$
    \ENDFOR
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
" %}


{% include pseudocode.html id="2" code="
\begin{algorithm}
\caption{Put your caption here}
\begin{algorithmic}

\Procedure{Roy}{$a,b$}       \Comment{This is a test}
    \State System Initialization
    \State Read the value
    \If{$condition = True$}
        \State Do this
        \If{$Condition \geq 1$}
        \State Do that
        \ElsIf{$Condition \neq 5$}
        \State Do another
        \State Do that as well
        \Else
        \State Do otherwise
        \EndIf
    \EndIf

    \While{$something \not= 0$}  \Comment{put some comments here}
        \State $var1 \leftarrow var2$  \Comment{another comment}
        \State $var3 \leftarrow var4$
    \EndWhile
\EndProcedure

\end{algorithmic}
\end{algorithm}
" %}



{% include pseudocode.html id="5" code="
\begin{algorithm}
\caption{SLAM}
\begin{algorithmic}
\PROCEDURE{SLAM}{$X_{t-1}, u_t, z_t$}
    \STATE $\bar{X}_t = X_t = \empty$
    \FOR{$m = 1$ \TO $M$}
        \STATE $x_t^{[k]} = $ \CALL{MotionUpdate}{$u_t, x_{t-1}^{[k]}$}
        \STATE $w_t^{[k]} = $ \CALL{SensorUpdate}{$z_t, x_{t}^{[k]}$}
        \STATE $m_t^{[k]} = $ \CALL{UpdateOccupancyGrid}{$z_t, x_{t}^{[k]}, m_{t-1}^{[k]}$}
        \STATE $\bar{X}_t = \bar{X}_t + \left < x_{t}^{[k]}, w_{t}^{[k]} \right >$
    \ENDFOR
    \FOR{$k = 1$ \TO $M$}
        \STATE draw $i$ with probability $w_t^{[i]}$
        \STATE add $\left < x_t^{[i]}, m_t^{[i]} \right >$ \TO $X_t$
    \ENDFOR
    \RETURN $X_t$
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
" %}

