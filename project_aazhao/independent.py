import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import mmread
import random
import os
import numpy as np

# create each of the 3 networks
os.makedirs("plots", exist_ok=True)
higgs = nx.read_edgelist("network_structure/higgs-social_network.edgelist", nodetype=int, create_using=nx.DiGraph)
twitter = nx.read_edgelist("network_structure/twitter_combined.txt", nodetype=int, create_using=nx.DiGraph)
A = mmread("network_structure/soc-twitter-follows/soc-twitter-follows.mtx").tocsr()
soc = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)

# this function implements the independent cascade with decay spreading process
# graph is the network to run it on
# seed_negative is the initial negative nodes
# seed_positive is the initial positive nodes
# p0 is the base probability
# alpha is the decay factor
# max_steps is the max amount of timesteps for the spreading process
def independent_cascade_decay(graph, seed_negative, seed_positive, p0, alpha, max_steps):
    # initialize positive, negative, and neutral nodes
    state = {node: "neutral" for node in graph.nodes}
    for n in seed_negative:
        state[n] = "negative"
    for p in seed_positive:
        state[p] = "positive"

    # initialize newly active nodes to determine next infections
    newly_active = set(seed_negative) | set(seed_positive)

    # track total # of positive and negative nodes at each timestep
    cumulative_pos = []
    cumulative_neg = []

    # track total # of positive and negative tweets at each timestep
    tweets_pos = []
    tweets_neg = []
    cum_tweets_pos = []
    cum_tweets_neg = []

    t = 0
    # iterate until max steps or no newly active nodes
    while newly_active and t < max_steps:
        # track node to pos/neg mapping and newly active nodes
        next_state_updates = {}
        next_newly_active = set()
        # probability of activation at time t
        p_t = p0 * (alpha ** t)

        # iterate through all followers of newly active nodes
        # once active, each node gets 1 chance to activate each neighbor
        for node in newly_active:
            for neighbor in graph.predecessors(node):
                # ignore prev activated or newly activated neighbors
                if state[neighbor] != "neutral" or neighbor in next_state_updates:
                    continue

                # chance p_t of activaton
                if random.random() < p_t:
                    # determine total # of activated following
                    parents = list(graph.successors(neighbor))
                    neg_count = sum(state[u] == "negative" for u in parents)
                    pos_count = sum(state[u] == "positive" for u in parents)
                    total = neg_count + pos_count

                    # activate based on majority of active following or random tiebreak
                    if total == 0:
                        chosen = random.choice(["negative", "positive"])
                    else:
                        if neg_count > pos_count:
                            chosen = "negative"
                        elif pos_count > neg_count:
                            chosen = "positive"
                        else:
                            chosen = random.choice(["negative", "positive"])
                    next_state_updates[neighbor] = chosen

        # update activated nodes with positive/negative & prepare next round
        for node, new_state in next_state_updates.items():
            state[node] = new_state
            next_newly_active.add(node)
        newly_active = next_newly_active

        # count total # of positive and negative nodes
        cumulative_pos.append(sum(state[n] == "positive" for n in graph.nodes))
        cumulative_neg.append(sum(state[n] == "negative" for n in graph.nodes))
        # count total # of tweets at this timestep
        pos_tweets = sum(state[n] == "positive" for n in graph.nodes)
        neg_tweets = sum(state[n] == "negative" for n in graph.nodes)

        # count total # of positive/negative tweets so far
        tweets_pos.append(pos_tweets)
        tweets_neg.append(neg_tweets)
        cum_tweets_pos.append(sum(tweets_pos))
        cum_tweets_neg.append(sum(tweets_neg))
        t += 1

    return (cumulative_pos, cumulative_neg, cum_tweets_pos, cum_tweets_neg)

# helper function for python lists to pad array for averaging
def pad_to_length(arr, target_len):
    if len(arr) == target_len:
        return arr
    # fill extra space with last elem
    return arr + [arr[-1]] * (target_len - len(arr))

graphs = {
    "Higgs": higgs,
    "Twitter": twitter,
    "Soc-Twitter-Follows": soc
}

# parameters
p0_vals = [0.05, 0.33] #base prob
alpha_vals = [0.9, 0.75] #decay factor
steps_vals = [25] # max steps
starting_nums = [2500, 7500, "influencer"] # seed nodes, influencer is top 100 deg nodes

# runs to average over
RUNS = 15

# iterate through all combinations of parameters
for p0_val in p0_vals:
    for alpha_val in alpha_vals:
        for steps_val in steps_vals:
            for starting_num in starting_nums:
                # maps network to # of pos/neg nodes & tweets 
                results = {}

                # run process on each graph
                for name, G in graphs.items():
                    # store # of pos/neg nodes & tweets 
                    avg_cum_pos = None
                    avg_cum_neg = None
                    avg_total_pos = None
                    avg_total_neg = None
                    for _ in range(RUNS):
                        # generate seed nodes
                        if starting_num == "influencer":
                            all_nodes = list(G.nodes)
                            top_nodes = sorted(all_nodes, key=lambda x: G.degree[x], reverse=True)[:100]
                            influencer_sample = random.sample(top_nodes, 100)
                            # split top 100 degree nodes randomly & evenly
                            seed_neg = influencer_sample[:50]
                            seed_pos = influencer_sample[50:]
                        else:
                            # mutually exclusive sets of pos/neg seeds
                            all_nodes = list(G.nodes)
                            seed_neg = random.sample(all_nodes, starting_num)
                            remaining_nodes = list(set(all_nodes) - set(seed_neg))
                            seed_pos = random.sample(remaining_nodes, min(starting_num, len(remaining_nodes)))
                        # run spreading process
                        (pos_counts, neg_counts,cum_pos, cum_neg) = independent_cascade_decay(G, seed_neg, seed_pos,p0_val, alpha_val, steps_val)

                        # pad arrays for processing
                        run_len = len(cum_pos)
                        if avg_cum_pos is None:
                            max_len = run_len
                        else:
                            max_len = len(avg_cum_pos)
                        # if current run is longer than prev, extend prev
                        if run_len > max_len:
                            avg_cum_pos = np.pad(avg_cum_pos, (0, run_len - max_len), mode='edge')
                            avg_cum_neg = np.pad(avg_cum_neg, (0, run_len - max_len), mode='edge')
                            avg_total_pos = np.pad(avg_total_pos, (0, run_len - max_len), mode='edge')
                            avg_total_neg = np.pad(avg_total_neg, (0, run_len - max_len), mode='edge')
                            max_len = run_len
                        # pad current run
                        cum_pos = pad_to_length(cum_pos, max_len)
                        cum_neg = pad_to_length(cum_neg, max_len)
                        pos_counts = pad_to_length(pos_counts, max_len)
                        neg_counts = pad_to_length(neg_counts, max_len)
                        # update running totals
                        if avg_cum_pos is None:
                            avg_cum_pos = np.array(cum_pos, dtype=float)
                            avg_cum_neg = np.array(cum_neg, dtype=float)
                            avg_total_pos = np.array(pos_counts, dtype=float)
                            avg_total_neg = np.array(neg_counts, dtype=float)
                        else:
                            avg_cum_pos += np.array(cum_pos)
                            avg_cum_neg += np.array(cum_neg)
                            avg_total_pos += np.array(pos_counts)
                            avg_total_neg += np.array(neg_counts)

                    # average the results
                    avg_cum_pos /= RUNS
                    avg_cum_neg /= RUNS
                    avg_total_pos /= RUNS
                    avg_total_neg /= RUNS
                    results[name] = (avg_cum_pos, avg_cum_neg, avg_total_pos, avg_total_neg)

                final_error = 0
                plt.figure(figsize=(8, 6))
                # determine & plot positive tweet to total tweet ratio
                # determine absolute error vs baseline 0.5 (even # of pos/neg tweets)
                for name, (cum_pos, cum_neg, _, _) in results.items():
                    ratios = [(cum_pos[i] / (cum_pos[i] + cum_neg[i])) if (cum_pos[i] + cum_neg[i]) > 0 else 0 for i in range(len(cum_pos))]
                    plt.plot(range(len(ratios)), ratios, marker='o', label=name)
                    final_error += abs(ratios[-1] - 0.5)
                # average the absolute error 
                error = final_error / len(results)
                print(f"params: {p0_val} {alpha_val} {steps_val} {starting_num} error: {error}")
                # plot
                plt.xlabel("Timestep")
                plt.ylabel("Tweet Positivity Ratio")
                plt.title(f"p0={p0_val}, α={alpha_val}, seeds={starting_num}, error={error:.4f}")
                plt.legend()
                plt.ylim(0, 1)
                plt.axhline(0.5, linestyle='--', color='black')
                plt.axhspan(0.5, 1, facecolor='lightgreen', alpha=0.3) # more pos than neg
                plt.axhspan(0, 0.5, facecolor='lightcoral', alpha=0.3) # less pos than neg

                # save plot
                filename = f"plots/ratio_p{p0_val}_a{alpha_val}_n{starting_num}.png"
                filename = filename.replace('.', '_')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()

                # plot total # of positive & negative tweets
                for name, (_, _, avg_total_pos, avg_total_neg) in results.items():
                    plt.plot(range(len(avg_total_pos)), avg_total_pos, marker='o', label=f"{name} (pos)")
                    plt.plot(range(len(avg_total_pos)), avg_total_neg, marker='o', label=f"{name} (neg)")
                # plot
                plt.xlabel("Timestep")
                plt.ylabel("Total Positive/Negative Tweets")
                plt.title(f"p0={p0_val}, α={alpha_val}, seeds={starting_num}, error={error:.4f}")
                plt.legend()
                
                # save plot
                filename = f"plots/tweets_p{p0_val}_a{alpha_val}_n{starting_num}.png"
                filename = filename.replace('.', '_')
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()