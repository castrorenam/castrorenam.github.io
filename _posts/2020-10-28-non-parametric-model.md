---
layout: cpost
title: "Reading notes: Non-parametric model"
date: 20-10-28
categories: jekyll update
---


# In construction

## What

Reading notes about the non-parametric stistical model for latent representation produced by
compressive autoencoders. Specifically, the modeling described in the paper [Variational Image Compression with a Scale Hyperprior](https://arxiv.org/abs/1802.01436).


## Objective

Given a set of observation sampled from an unknown distribution, we want to fit a non parametric distribution model to the source distribution.

### Cdf as composition of functions

In order to optimize variational autoencoders for image compression, one is required to model the probability of the latent representation
**z**. A straightforward approach assumes the latent representation follows a factorized distribution:

\begin{equation}
    P_{ \tilde{\mathbf{y}}| \boldsymbol{\psi}^{(i)} } = \prod_{i}  P_{ \mathbf{y}_i| \boldsymbol{\psi}^{(i)}} \( \mathbf{y}_i \ast u(-1/2, 1/2); \boldsymbol{\psi}^{(i)} \)   
\end{equation}

The authors propose the following (learnable) composition of function for the cdf *c* of the latent:

\begin{equation}
    c = f_K \circ f_{K-1} \circ \cdot\cdot\cdot \circ f_1
\end{equation}
where for sake of generality $f_k: \mathbb{R}^{d_k} \rightarrow \mathbb{R}^{r_k}$. 

Therefore, the pdf *p* would be:

\begin{equation}
    p = f_{K}^{\prime} \circ f_{K-1}^{\prime} \circ \cdot\cdot\cdot \circ f_1^{\prime}
\end{equation}
where $f_{K}^{\prime}$ are Jacobian matrices

For univariate distributions the domain of $f_1$ and range of $f_K$ need to be 1-D. To make sure *c* is a valid cdf one need:

1. $p(x) \geq 0 $.

2. $c(-\infty) = 0$ and $c(\infty) = 1$

To satisfy condition 1, one need every $f_{k}^{\prime}$ in Equation 3 to be positive. This justifies the functions used by the authors.

To satisfy condition 2, the authors choose the sigmoid for $f_K$.


#### Parametric functions $f_k$

The $f_k$ for $1 \leq k  < K$ is defined as:
\begin{equation}
    f_k(\mathbf{x}) = g_k(\mathbf{H}^{(k)} \mathbf{x} + \mathbf{b}^{(k)})
\end{equation}
where $\mathbf{H}$s and $\mathbf{b}$s are learnable parameters, matrices and vectors respectively.

The nonlinearity $g_k$ is defined as:

\begin{equation}
    g_k(\mathbf{x}) = \mathbf{x} + \mathbf{a}^{(k)} \odot tanh(\mathbf{x})
\end{equation}
where $\mathbf{a}$s are learnable vectors.


To guarantee the $f_k^{\prime}$s are non-negative, the authors defined $\mathbf{H}$s and $\mathbf{a}$s as follows:

\begin{equation}
    \mathbf{H}^{(k)} = softplus(\hat{\mathbf{H}}^{(k)}) 
\end{equation}

\begin{equation}
    \mathbf{a}^{(k)} = tanh(\hat{\mathbf{a}}^{(k)})
\end{equation}
where $\hat{\mathbf{H}}$s and $\hat{\mathbf{a}}$s are in fact the learnable parameters.


The $f_K$ is simply the sigmoid aplied after linear transformation:

\begin{equation}
    f_K(\mathbf{x}) = sigmoid(\mathbf{H}^{(K)} \mathbf{x} + \mathbf{b}^{(K)})
\end{equation}

### Transformation view

Another way to see the proposed model for latent distribution is in terms of applying a transformation to a random variable. One may attemp to find a transform to be applied to the input random variable $\mathbf{z}$ of unknown distribution to random variable of known distribution. 

In probabilities textbooks, one usually encounter the problem of determining the pdf of a random variable resulted from a transformation $\mathbf{y} = f(\mathbf{x})$ from an input random $\mathbf{x}$ with known pdf $p(\mathbf{x})$. In such cases, the pdfs are related by:

\begin{equation}
    p(\mathbf{y}) = p(\mathbf{x}) |J|
\end{equation}
where $|J|$ is absolute value of the Jacobian determinant.

Note that the last function applied in Equation 2 is the sigmoid function, which is known to be the cdf function of the logistic distribution. One can view the chain of functions applied to the input variable as a transformation to change it into a logistic-distributed random variable.

\begin{equation}
    c(x) = sigmoid(x)
\end{equation}

where,
\begin{equation}
    x =  \mathbf{H}^{K} (f_{K-1} \circ \cdot\cdot\cdot \circ f_1) + \mathbf{b}^{K}
\end{equation}

Therefore, the chain of transformation is in fact turning the input variable of unknown distribution into a standard logistic distribution variable, which is in turn used to estimate the entropy of the latent. 


Suppose we have a set of samples {$z_1, z_2, ...$} sampled from a unknown distribution.

\begin{equation}
    z \sim unknow
\end{equation}


We want to find a transform $x = T(z, \psi)$ parametrized by $\psi$ such that:

\begin{equation}
    x  \sim logistic
\end{equation}


### Objective function

To drive the learning of the model parameters, the loss function is simply the entropy of $\mathbf{z}$ with respect to the fitted model. Thus, it is expected the model will be driven to provide accurate probability estimates:

\begin{equation}
    Loss = -\mathbb{E}_{p(\mathbf{z})} \lbrace log \ p(\mathbf{z}) \rbrace
\end{equation}


As the pdf is assumed to be factorized: 

\begin{equation}
    Loss = -\mathbb{E}_{p(\mathbf{z})} \lbrace log \ \prod_i p(z_i) \rbrace
\end{equation}


In practise, we approximate the expectation by averaging over a large data set of samples. 

\begin{equation}
    Loss = - \frac{1}{N} \sum_s^N  log \ p(\mathbf{z})
\end{equation}


```bash
     conda env list
```

































