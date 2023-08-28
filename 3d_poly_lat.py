import numpy as np
import matplotlib.pyplot as plt

# we want to sample SAWs on a 3d lattice first. To do this, treat it like sampling from a Boltzmann distribution and make a strong repulsive interaction energy?

sigma = 10 #interaction energy for overlapping monomers, in units of K_b T. Probability of overlap should scale like exp(-sigma)

step2vec = {0:np.array([0,1,0]),1:np.array([0,-1,0]),2:np.array([-1,0,0]),3:np.array([1,0,0]),4:np.array([0,0,1]),5:np.array([0,0,-1])}
    # Convention: "steps" is an (N-1)x1 array where N is the number of monomers in the chain. Each step is a number from 0-5. 0: up (y+), 1: down(y-), 2: left(x-), 3:right(x+), 4:out of the page(z+), 5:into the page(z-)

def target_dist(steps,initpos = 0): #"steps" is the array of steps in the lattice SAW. We must convert from "steps" to an array of positions, and from there evaluate the probability distribution (here we'll treat it as Boltzmann for now).
    
