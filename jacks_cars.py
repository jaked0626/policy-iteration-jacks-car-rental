#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)    
# 2022 Jake Underland (jakez0626@gmail.com)
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Example 4.2: Jack’s Car Rental Jack manages two locations for a nationwide car
# rental company. Each day, some number of customers arrive at 
# each location to rent cars. If Jack has a car available, he rents it out and is 
# credited $10 by the national company. If he is out of cars at that location, 
# then the business is lost. Cars become available for renting the day after they 
# are returned. To help ensure that cars are available where they are needed, 
# Jack can move them between the two locations overnight, at a cost of $2 per 
# car moved. We assume that the number of cars requested and returned at 
# each location are Poisson random variables, meaning that the probability that 
# the number is n is (λ^n/n!) * exp(-λ) where λ is the expected number. Suppose λ is 3 
# and 4 for rental requests at the first and second locations and 3 and 2 for 
# returns. To simplify the problem slightly, we assume that there can be no more
#  than 20 cars at each location (any additional cars are returned to the nationwide
#  company, and thus disappear from the problem) and a maximum of five cars can be moved
#  from one location to the other in one night. We take the GAMMA rate to be γ = 0.9
#  and formulate this as a continuing finite MDP, where the time steps are days,
#  the state is the number of cars at each location at the end of the day, and 
# the actions are the net numbers of cars moved between the two locations overnight.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

### PARAMETERS FOR PROBLEM ###

CAR_RENTAL_REWARD = 10
CAR_MOVE_REWARD = -2
MAX_CARS = 20
MAX_MOVE = 5
# action space
ACTIONS = list(range(- MAX_MOVE, MAX_MOVE + 1))
# Lambdas for requests and returns at stores one and two
REQUEST_ONE_LAMBDA = 3
RETURN_ONE_LAMBDA = 3
REQUEST_TWO_LAMBDA = 4 
RETURN_TWO_LAMBDA = 2
# discount rate
GAMMA = 0.9
# Upper bound for poisson distribution above which probability is 0
POISSON_UPPER_BOUND = 11


class JacksCarRental:
    def __init__(self):
        self.initialization()
    
    ################ PREPARE ENVIRONMENT ###########################

    def initialization(self):
        """
        Corresponds to the initialization step in the policy iteration algorithm. 
        V(s) ∈ R and π(s) ∈ A(s) ∀ s ∈ S
        """
        self._poisson_prob = {}
        # matrix with values of states V(s)
        self.value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        # policy matrix 
        self.policy = self.value.copy().astype(int)
        # state transition probabilites and expected rewards per store
        self._probs_1, self._rewards_1 = self.precompute_dynamics(
            REQUEST_ONE_LAMBDA, RETURN_ONE_LAMBDA)
        self._probs_2, self._rewards_2 = self.precompute_dynamics(
            REQUEST_TWO_LAMBDA, RETURN_TWO_LAMBDA)

    def precompute_dynamics(self, lambda_requests, lambda_returns):
        """
        Precomputes the model dynamics for efficiency: the expected reward and the 
        state transition probabilities for each store. 

        Inputs:
            lambda_requests (int): Lambda for the probability distribution of requests
            lambda_returns (int): Lambda for the probability distribution of returns

        Returns:
            P (2D Numpy Array[int]): Matrix containting state transition probabilities
            R (Numpy Array[int]): Expected reward per state. 
        """
        P, R = {}, {}

        # go through all possible rental requests
        # probability of n > POISSON_UPPER_BOUND being requested is 0
        for requested_cars in range(POISSON_UPPER_BOUND): 
            request_prob = self.poisson_probability(requested_cars, lambda_requests)
            # number of cars at store (the state) when rental requests are made
            for state_n in range(MAX_CARS + MAX_MOVE + 1):
                # reward R sums all rewards with their respective probability per state state_ncars
                satisfied_requests = min(requested_cars, state_n)
                R[state_n] = R.get(state_n, 0) + CAR_RENTAL_REWARD * request_prob * satisfied_requests
                # now calculate transition probabilities
                # for this we need both request and dropoff information
                for returns in range(POISSON_UPPER_BOUND):
                    return_prob = self.poisson_probability(
                        returns, lambda_returns)
                    new_state_n = max(0, min(MAX_CARS, state_n + returns - satisfied_requests))
                    P[(state_n, new_state_n)] = P.get((state_n, new_state_n), 0) \
                                                + request_prob * return_prob
        return P, R
    
    ######################### UTILITIES ###############################

    def step(self, state, action):
        """
        Run one timestep of the model.
        Returns s' given s and a. 

        Inputs:
            state (Tuple[int, int]): A tuple storing the number of cars 
                                     at stores 1 and 2 respectively
            action (int): The number of cars to be moved from store 1 to 2

        Returns:
            (Tuple[int, int]): The new state of the system
        """
        cars_one = int(state[0] - action)
        cars_two = int(state[1] + action)
        new_state = (cars_one, cars_two)
        return new_state
    
    def get_reward(self, state):
        """
        Computes the reward for the given state.

        Inputs:
            state (Tuple[int, int]): A tuple storing the number of cars 
                                     at stores 1 and 2 respectively

        Returns:
            (float): The expected reward for the given state
        """
        return self._rewards_1[state[0]] + self._rewards_2[state[1]]

    def get_transition_probability(self, state, new_state):
        """
        Returns p(s' | s). 
        Inputs:
            state (Tuple[int, int]): A tuple storing the number of cars 
                                     at stores 1 and 2 respectively (s)
            new_state (Tuple[int, int]): new state (s')

        Returns:
            (float): Probability of environment transitioning from s to s'
        """
        try: 
            return self._probs_1[(state[0], new_state[0])] * self._probs_2[(state[1], new_state[1])]
        except KeyError: # necessary due to upper bound for poisson
            return 0

    def get_valid_action(self, state, action):
        """
        Returns all valid (possible) actions. For example, stores cannot transfer
        more cars than they currently have. 

        Inputs:
            state (Tuple[int, int]): A tuple storing the number of cars 
                                     at stores 1 and 2 respectively (s)
            action (int): The number of cars to be moved

        Returns:
            (int): a feasible number of cars to be moved from store 1 to 2
        """
        cars_at_1, cars_at_2 = state
        # Jack can't move more cars than he has available
        action = max(-cars_at_2, min(action, cars_at_1))
        # Jack can move at most 5 cars
        action = max(-MAX_MOVE, min(MAX_MOVE, action))
        return action

    def get_available_actions(self, state):
        """
        Returns all possible actions (not just valid)

        Inputs:
            state (Tuple[int, int]): A tuple storing the number of cars 
                                     at stores 1 and 2 respectively (s)
        Returns:
            (List[int]): list of actions 
        """
        return list(range(max(-MAX_CARS, - state[1]), min(MAX_CARS, state[0]) + 1))

    def poisson_probability(self, n, lam):
        """
        Computes the probability that the number drawn from a 
        poisson distribution is `n`, given a lamdda of `lam`.
        $p = e^(-λ) * (λ^n / n!)$

        Inputs:
            n (int): the number expected to be drawn from the distribution
            lam (int): the λ parameter of the poisson distribution

        Returns:
            (float): the probability that the number is `n`
        """
        key = (n, lam)
        if key not in self._poisson_prob:
            self._poisson_prob[key] = poisson.pmf(n, lam)
        return self._poisson_prob[key]

    def render(self, iteration, policy_stable):
        """
        Plots the current value table and the current policy
        """
        # plot value table
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.heatmap(self.value, cmap="gray_r", ax=ax[0])
        ax[0].set_ylim(0, MAX_CARS+1)
        ax[0].set_title("Value table V_π")
        # plot policy
        cmaplist = [plt.cm.RdBu(i) for i in range(plt.cm.RdBu.N)]
        dRbBu = matplotlib.colors.LinearSegmentedColormap.from_list(
            'dRdBu', cmaplist, plt.cm.RdBu.N)
        sns.heatmap(self.policy, vmin=-MAX_MOVE, vmax=MAX_MOVE, cmap=dRbBu,
                    ax=ax[1], cbar_kws={"ticks": ACTIONS, "boundaries": ACTIONS})
        ax[1].set_ylim(0, MAX_CARS+1)
        ax[1].set_title("Policy π")
        if policy_stable:
            iteration = f"{iteration} : Policy Converged"
        fig.canvas.set_window_title(f"Results of iteration {iteration}")
        plt.show()
        return fig, ax
    
    ############ POLICY ITERATION (CONSULT FIGURE 4.3 OF SUTTON & BARTO) #############

    def bellman_expectation(self, state, action):
        """
        Solves the bellman expectation equation for given state
        V(s) = p(s, r | s' π(s)) * (R(s) + γ * V(s'))

        Inputs:
            state (Tuple[int, int]): A tuple storing the number of cars 
                                     at stores 1 and 2 respectively (s)
            action (int): The number of cars to be moved from store 1 to 2

        Returns:
            (float): the renewed value V(s) of the current state pair
        """
        a = self.get_valid_action(state, action)
        s = self.step(state, a)
        r = self.get_reward(s)

        state_value = CAR_MOVE_REWARD * abs(a)
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                sprime = (i, j)
                p = self.get_transition_probability(s, sprime)
                state_value += p * (r + GAMMA * self.value[i, j])
        return state_value

    def policy_evaluation(self, theta=1e-3):
        """
        Renews the value table for the current policy using iterative
        policy evaluation.

        Inputs: 
            theta (float): tolerated margin of error
        """
        while True:
            delta = 0
            # for each state s ∈ S
            for cars_A in range(MAX_CARS + 1):
                for cars_B in range(MAX_CARS + 1):
                    # s
                    state = (cars_A, cars_B)
                    # a <- π(s)
                    action = self.policy[state]
                    # v <- V(s)
                    v = self.value[state]
                    # V(s) <- Σ_s',r p(s', r | s, π(s)) * (R(s) + γ * V(s'))
                    self.value[state] = self.bellman_expectation(state, action)
                    # Δ <- max(Δ, |v - V(s)|)
                    delta = max(delta, abs(v - self.value[state]))
            print("\t\tValue delta {:.5f}\t\t ".format(delta), end="\r")
            if delta < theta: break
        
    def policy_improvement(self):
        """
        Makes one step of policy improvement following a greedy policy.
        For each state of the model, it iterates through all the feasible 
        actions and finds the greediest one.

        Returns:
            policy_stable (bool): True if the policy is stable (unchanged).
        """
        policy_stable = True
        # for each state s ∈ S
        for available_A in range(MAX_CARS + 1):
            for available_B in range(MAX_CARS + 1):
                # s
                state = (available_A, available_B)
                # a <- π(s)
                a = self.policy[state]
                # π(s) <- argmax_a Σ_s',r p(s', r | s, a) [R(s) + γV(s')]
                max_value = float("-inf")
                for action in self.get_available_actions(state):
                    try:
                        value = self.bellman_expectation(state, action)
                    except:
                        print(state, self.get_available_actions(state))
                    if value > max_value:
                        max_value = value
                        # update π(s)
                        self.policy[state] = action
            # if a ≠ π(s), then policy-stable <- false
            if a != self.policy[state]: policy_stable = False
        return policy_stable

    def policy_iteration(self, plot=False):
        """
        Computes the optimal policy π* using policy iteration.

        Inputs:
            plot (bool): boolean indicating whether results should be plotted.

        Returns:
            self.policy (Numpy 2D Array): The optimal policy matrix
        """
        iter = 1
        iterate = True

        # Repeat
        while iterate:
            # log
            print(f"Iteration number {iter}:")

            # policy evaluation to update the value table
            print(f"\tEvaluating policy {iter}")
            self.policy_evaluation()

            # policy improvement to update the current policy, based on the new value table
            print(f"\tImproving policy {iter}")
            policy_stable = self.policy_improvement()

            # has π converged to π*?
            if policy_stable:
                print("Policy is stable. π converged to π*")
                iterate = False
            
            # plot
            if plot:
                self.render(iter, policy_stable)
            
            iter += 1
        
        print(self.policy)
        return self.policy


if __name__ == "__main__":
    sim_problem = JacksCarRental()
    sim_problem.policy_iteration(True)