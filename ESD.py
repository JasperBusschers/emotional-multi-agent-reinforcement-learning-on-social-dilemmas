import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import random

social_dilemma = np.zeros([2, 2])
social_dilemma[0, 0] = 3
social_dilemma[0, 1] = 0
social_dilemma[1, 0] = 5
social_dilemma[1, 1] = 1


class Grid:
    def __init__(self, node_number):
        self.G = nx.grid_graph(dim=[int(math.sqrt(node_number)), int(math.sqrt(node_number))], periodic=False)

    def edges(self):
        return self.G.edges

    def nodes(self):
        return self.G.nodes

    def get_neighbors(self, v):
        return self.G.neighbors(v)

    def average_degree(self):
        degrees = [val for (node, val) in self.G.degree()]
        return np.mean(degrees)


class Barabasi_Albert:
    def __init__(self, node_number):
        self.G = nx.complete_graph(5)
        new_node = 5
        while len(self.G.nodes) < node_number:
            choosed = np.random.choice(self.G.nodes, 1, p=self.calculate_probabilities())
            self.G.add_node(new_node)
            self.G.add_edge(choosed[0], new_node)
            new_node += 1

    def choose(self, distribution):
        return np.random.choice(self.G.nodes, 1, p=distribution)

    def calculate_probabilities(self):
        degrees = [val for (node, val) in self.G.degree()]
        total = np.sum(degrees)
        probs = []
        i = 0
        for node in self.G.nodes:
            prob = self.G.degree(node)/total
            probs.insert(i, prob)
            i += 1

        return probs

    def edges(self):
        return self.G.edges

    def nodes(self):
        return self.G.nodes

    def get_neighbors(self, v):
        return self.G.neighbors(v)

    def average_degree(self):
        degrees = [val for (node, val) in self.G.degree()]
        return np.mean(degrees)


class Watts_Strogatz:
    def __init__(self, node_number, k, p):
        self.G = nx.watts_strogatz_graph(node_number, k, p, seed=None)

    def edges(self):
        return self.G.edges

    def nodes(self):
        return self.G.nodes

    def get_neighbors(self, v):
        return self.G.neighbors(v)

    def average_degree(self):
        degrees = [val for (node, val) in self.G.degree()]
        return np.mean(degrees)


class prisonersGame():
    def __init__(self, network, dilemma, policy, number_of_actions=2, number_of_policies=2, number_of_emotions=4, core_coeff=0.2, secondary_coeff=0.2, epsilon=0.1, alpha_losing=0.1, alpha_winning=0.04):
        # Type of network
        self.G = network
        # Playing Prisoners' Dilemma
        self.dilemma = dilemma
        # Type of policy
        self.policy = policy

        # Q table for the WF agents
        self.Q_table_WF = np.random.uniform(low=0.75, high=1.5, size=(len(network.nodes()), number_of_emotions, number_of_actions))
        # Q table for the FW agents
        self.Q_table_FW = np.random.uniform(low=0.75, high=1.5, size=(len(network.nodes()), number_of_emotions, number_of_actions))
        # Q table for the R agents
        self.Q_table_Rational= np.random.uniform(low=0.75, high=1.5, size=(len(network.nodes()), number_of_actions))
        # Table to store the emotions of all the nodes in Graph
        self.emotions = np.random.randint(low=0, high=4, size=len(network.nodes()))
        # Exploration rate
        self.epsilon = epsilon
        # ???
        self.alpha_winning = alpha_winning
        self.alpha_losing = alpha_losing
        # Learning rate
        self.alpha = 0.5

    # Wellbeing appraisal approaches
    # Absolute value-based approach
    def W_abs(self, Rt, m):
        return (2 * Rt - (m * (self.dilemma[1, 0] - self.dilemma[0, 1]))) / (m * (self.dilemma[1, 0] - self.dilemma[0, 1]))

    # Variance-based approach
    def W_var(self, Rt, Rt2, m):
        return ((Rt2 - Rt) / (m * (self.dilemma[1, 0] - self.dilemma[0, 1])))

    # Aspiration-based approach
    def W_asp(self, Rt, At, h, m):
        return (math.tanh(h * (Rt / m - At)))

    # Calculate Aspiration level for Aspiration-based approach
    def calc_A(self, Rt, At, beta, m):
        return ((1 - beta) * At + beta * Rt / m)

    def emotion_defivation_model(self, F, W, node):
        # Fairness-Wellbeing (FW) Emotion Derivation Function
        if self.policy == 0:
            if F >= 0:
                self.emotions[node] = 0  # happy = 0
                E = self.f(F) * self.g(W)
            else:
                if W > 0:
                    self.emotions[node] = 3  # fearful = 3
                    E = self.f(-F) * self.g(W)
                else:
                    self.emotions[node] = 2  # angry = 2
                    E = self.f(-F) * self.g(-W)

        # Wellbeing-Fairness (WF) Emotion Derivation Function
        elif self.policy == 1:
            if W > 0:
                self.emotions[node] = 0  # happy = 0
                E = self.f(W) * self.g(F)
            else:
                self.emotions[node] = 1  # sad = 1
                E = self.f(-W) * self.g(F)
        return E

    def choose_action(self, node, emotion):

        if random.random() < self.epsilon:
            action = random.randint(0, 1)
        else:
            if self.policy == 0:
                # Policy Fairness-Wellbeing
                action = np.argmax(self.Q_table_FW[node, emotion, :])
            elif self.policy == 1:
                # Policy Wellbeing-Fairness
                action = np.argmax(self.Q_table_WF[node, emotion, :])
            elif self.policy == 2:
                # Policy Rational
                action = np.argmax(self.Q_table_Rational[node, :])
        return action

    def policyPalying(self, episodes, M, approach_W):
        # Scalable Parameter
        h = 10
        # Learning rate
        beta = 0.5
        total_rewards = []
        # To record the absolute value of reward every episode
        record_Abs = np.zeros([episodes, len(self.G.nodes())])
        # To record the aspiration level every episode
        aspiration_level = np.zeros([episodes + 1, len(self.G.nodes())])
        for i in range(0, episodes):
            single_reward = []
            for node in self.G.nodes():
                count = 0
                C_node = 0
                action_num_node = np.zeros(2)
                # To record the reward for each node
                rewards_node = []
                emotion_node = self.emotions[node]
                action_node = self.choose_action(node, emotion_node)
                for neighbor in self.G.get_neighbors(node):
                    action_num_neighbor = np.zeros([len(self.G.nodes()), 2])
                    emotion_neighbor = self.emotions[neighbor]
                    for m in range(0, M):
                        action_neighbor = self.choose_action(node, emotion_neighbor)
                        # Record the actions taken by neighbors
                        action_num_neighbor[neighbor, action_neighbor] += 1
                        # Record the actions taken by focal node
                        action_num_node[action_node] += 1

                        rewards_node.append(self.dilemma[action_node, action_neighbor])
                    count += 1
                    C_node += ((action_num_neighbor[neighbor, 0] - action_num_neighbor[neighbor, 1]) / M)

                abs_wealth = np.sum(rewards_node)
                record_Abs[i, node] = abs_wealth / count

                F_node = C_node / count * ((action_num_node[0] - action_num_node[1]) / M)

                # Wellbeing Appraisal using Absolute value-based approach
                if approach_W == "abs":
                    W_node = self.W_abs(abs_wealth / count, M)

                elif approach_W == "var":
                    # Wellbeing Appraisal using variance-based approach
                    if i >= 1:
                        W_node = self.W_var(record_Abs[i - 1, node], abs_wealth / count, M)
                    else:
                        W_node = self.W_abs(abs_wealth / count, M)

                elif approach_W == "asp":
                    # Wellbeing Appraisal using aspiration-based approach
                    if i >= 1:
                        W_node = self.W_asp(abs_wealth / count, aspiration_level[i, node], h, M)
                        aspiration_level[i+1, node] = self.calc_A(abs_wealth / count, aspiration_level[i, node], beta, M)
                    else:
                        aspiration_level[0, node] = (self.dilemma[0, 0] + self.dilemma[0, 1] + self.dilemma[1, 0] + self.dilemma[1, 1]) / 4
                        W_node = self.W_asp(abs_wealth / count, aspiration_level[0, node], h, M)
                        aspiration_level[1, node] = self.calc_A(abs_wealth / count, aspiration_level[0, node], beta, M)
                print(W_node)
                E_node = self.emotion_defivation_model(F_node, W_node, node)
                # Update using Intrinsic reward
                Rint_node = self.instrinctive_reward(E_node)
                # self.update_est(emotion_node, self.policy, Rint_node)
                self.update_est(node, action_node, Rint_node, emotion_node)

                single_reward.append(np.mean(rewards_node))

            total_rewards.append(np.mean(single_reward))

        return total_rewards

    def update_est(self, node, action, reward, emotion):
        alpha = self.alpha
        if self.policy == 0:
            self.Q_table_FW[node, emotion, action] += alpha * (reward - self.Q_table_FW[node, emotion, action])
        elif self.policy == 1:
            self.Q_table_WF[node, emotion, action] += alpha * (reward - self.Q_table_WF[node, emotion, action])
        elif self.policy == 2:
            self.Q_table_Rational[node, action] += alpha * (reward - self.Q_table_Rational[node, action])


    # Calculate Rint with the Absolute value of Ex
    def instrinctive_reward(self, x):
        return np.abs(x)

    # Linear functions to map the value of Dx and Ix to [0,1]
    def f(self, Dx):
        return Dx

    def g(self, Ix):
        res = (Ix + 1) / 2
        return res

    def rationalPlaying(self, episodes):
        total_rewards = []
        rewards = []
        for i in range(0, episodes):
            single_reward = []
            for node in self.G.nodes():
                # To record the reward for each node
                rewards_node = []
                count = 0
                for neighbor in self.G.get_neighbors(node):
                    action_node = self.choose_action(node, emotion_node)
                    action_neighbor = self.choose_action(node, emotion_neighbor)
                    rewards_node.append(self.dilemma[action_node, action_neighbor])
                    count += 1

                self.update_est(node, action_node, Rint_node, emotion_node)
                single_reward.append(np.mean(rewards_node))

            total_rewards.append(np.mean(single_reward))

        return rewards


# graph = Watts_Strogatz(100, 4, 0.4)
graph = Barabasi_Albert(100)
# graph = Grid(100)
policy_FW = 0
policy_WF = 1
M = 3
avg_fw = np.zeros([200])
avg_wf = np.zeros([200])
# Testing Absolute value-based approach
for i in range(0, 100):
    game_FW = prisonersGame(graph, social_dilemma, policy_FW)
    rewards_FW = game_FW.policyPalying(200, M, "abs")
    game_WF = prisonersGame(graph, social_dilemma, policy_WF)
    rewards_WF = game_WF.policyPalying(200, M, "abs")
    avg_fw += rewards_FW
    avg_wf += rewards_WF

# Testing Variance-based approach
# game_FW = prisonersGame(graph, social_dilemma, policy_FW)
# rewards_FW = game_FW.policyPalying(200, M, "var")
# game_WF = prisonersGame(graph, social_dilemma, policy_WF)
# rewards_WF = game_WF.policyPalying(200, M, "var")
# rewards_Rational =

# Testing Aspiration-based approach
# game_FW = prisonersGame(graph, social_dilemma, policy_FW)
# rewards_FW = game_FW.policyPalying(200, M, "asp")
# game_WF = prisonersGame(graph, social_dilemma, policy_WF)
# rewards_WF = game_WF.policyPalying(200, M, "asp")

avg_fw = avg_fw/100
avg_wf = avg_wf/100

plt.plot(avg_fw, label="reward FW")
plt.plot(avg_wf, label="reward WF")
plt.title("Average reward for each policy")
plt.xlabel("Iteration")
plt.ylabel("Average Reward")
plt.legend()
plt.show()
