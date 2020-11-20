---
layout: post
title: "Reading notes: Non-parametric model"
date: 20-10-28
---


# What

Reading notes about the non-parametric stistical model for latent representation produced by
compressive autoencoders. Specifically, the modeling described in the paper [Variational Image Compression with a Scale Hyperprior](https://arxiv.org/abs/1802.01436).


## Cdf function  

In order to optimize variational autoencoders for image compression, one is required to model the probability of the latent representation
**z**. The authors propose the following composition of function for the cdf:





