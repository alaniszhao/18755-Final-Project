import networkx as nx
import random
from scipy.io import mmread
import math
import matplotlib.pyplot as plt
import numpy as np

def assign_heterogeneous_thresholds(G, alpha, beta):
    thresholds = {}

    for n in G.nodes():

        thresholds[n] = random.betavariate(alpha, beta)

    return thresholds

def threshold_cascade(G, positive, pos_thresholds, negative, neg_thresholds):

    # active nodes at t=0
    pos_tweet_history = []
    neg_tweet_history = []
    pos_node_history = []
    neg_node_history = []
    pos_tweet_count = 0
    neg_tweet_count = 0

    while True:
        new_pos = set(positive)
        new_neg = set(negative)

        pos_tweet_history.append(pos_tweet_count)
        neg_tweet_history.append(neg_tweet_count)
        pos_node_history.append(len(positive))
        neg_node_history.append(len(negative))

        # loop through all nodes
        for v in G.nodes():

            # infected nodes produce tweets at each timestep
            if v in positive:
                pos_tweet_count += 1
                continue 

            if v in negative:
                neg_tweet_count += 1
                continue 
            
            preds = list(G.successors(v))
            if not preds:
                continue 

            pos_neighbors = sum(1 for u in preds if u in positive)
            neg_neighbors = sum(1 for u in preds if u in negative)

            # node is influenced by higher percentage of neighbors
            if pos_neighbors > neg_neighbors or ((pos_neighbors == neg_neighbors) and (random.randint(0, 1) == 0)):
                theta = pos_thresholds[v]
                frac_active = pos_neighbors / len(preds)
                if frac_active >= theta:
                    new_pos.add(v)
            else:
                theta = neg_thresholds[v]
                frac_active = neg_neighbors / len(preds)
                if frac_active >= theta:
                    new_neg.add(v)

        if (new_neg == negative) and (new_pos == positive):
            break

        positive = new_pos
        negative = new_neg

    return pos_tweet_history, pos_node_history, neg_tweet_history, neg_node_history

def find_ratios(pos, neg):
    ratios = []
    time = max(len(pos), len(neg))

    assert(len(pos) == len(neg))
    assert(len(pos) == time)

    for t in range(time):
        if pos[t] == 0 or neg[t] == 0:
            ratios.append(0)
        else:
            ratios.append(pos[t] / (pos[t] + neg[t]))

    return ratios

# pads arrays to same length
def pad_array(array, length):
    if len(array) < length:
        array += [array[-1]] * (length - len(array))
    
    return array


def run_model(G, a, b, seed, influencer=False):

    
    pos_thresholds = assign_heterogeneous_thresholds(G, a, b)
    neg_thresholds = assign_heterogeneous_thresholds(G, a, b)

    pos_tweet_counts = []
    neg_tweet_counts = []
    pos_node_count = []
    neg_node_count = []

    for k in range(15):

        # initialize random seed set
        if influencer:
            degree_dict = dict(G.degree())
            sorted_degrees = sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)
            top100 = [n for n, d in sorted_degrees[:100]]
            random.shuffle(top100)
            pos_seed = set(top100[:seed])
            neg_seed = set(top100[seed:2*seed])

        else:
            nodes = list(G.nodes())
            random.shuffle(nodes)
            pos_seed = set(nodes[:seed])
            neg_seed = set(nodes[seed:2*seed])

        # run cascade
        pos_tweets, pos_nodes, neg_tweets, neg_nodes = threshold_cascade(G, pos_seed, pos_thresholds, neg_seed, neg_thresholds)
        pos_tweet_counts.append(pos_tweets)
        pos_node_count.append(pos_nodes)
        neg_tweet_counts.append(neg_tweets)
        neg_node_count.append(neg_nodes)
    
    assert(len(pos_tweets) == len(pos_nodes))
    t = max(len(run) for run in (pos_tweet_counts + neg_tweet_counts))

    avg_pos_tweets = np.mean(np.array([pad_array(run, t) for run in pos_tweet_counts]), axis=0)
    avg_neg_tweets = np.mean(np.array([pad_array(run, t) for run in neg_tweet_counts]), axis=0)
    avg_pos_nodes = np.mean(np.array([pad_array(run, t) for run in pos_node_count]), axis=0)
    avg_neg_nodes = np.mean(np.array([pad_array(run, t) for run in neg_node_count]), axis=0)

    ratios = find_ratios(avg_pos_tweets, avg_neg_tweets)

    return ratios, avg_pos_nodes, avg_neg_nodes

def mae(higgs, twitter, soc):

    higgs_mae = np.mean(np.abs(np.array(higgs) - ground_truth))
    twitter_mae = np.mean(np.abs(np.array(twitter) - ground_truth))
    soc_mae = np.mean(np.abs(np.array(soc) - ground_truth))
    return (higgs_mae + twitter_mae + soc_mae) / 3


if __name__ == "__main__":

    ground_truth = 0.5

    # load networks
    higgs = nx.read_edgelist("./networks/higgs-social_network.edgelist", create_using=nx.DiGraph())
    twitter = nx.read_edgelist("./networks/twitter_combined.txt", create_using=nx.DiGraph())
    data = mmread('./networks/soc-twitter-follows/soc-twitter-follows.mtx')
    soc = nx.DiGraph(data)

    parameters = [(2, 5), (2, 2), (5, 2)]
    seed_count = [2500, 7500]

    for a, b in parameters:
        for seed in seed_count:

            higgs_ratio, higgs_pos_nodes, higgs_neg_nodes = run_model(higgs, a, b, seed)
            twitter_ratio, twitter_pos_nodes, twitter_neg_nodes = run_model(twitter, a, b, seed)
            soc_ratio, soc_pos_nodes, soc_neg_nodes = run_model(soc, a, b, seed)

            time = range(max(len(higgs_ratio), len(twitter_ratio), len(soc_ratio)))

            # plot tweets
            mae_tweets = mae(higgs_ratio, twitter_ratio, soc_ratio)
            plt.figure(figsize=(8, 5))
            plt.fill_between(time, ground_truth, 0, color="red", alpha=0.2)
            plt.fill_between(time, ground_truth, 1, color="green", alpha=0.2)
            plt.plot(range(len(higgs_ratio)), higgs_ratio, marker='o', label='Higgs', color="blue")
            plt.plot(range(len(twitter_ratio)), twitter_ratio, marker='o', label='Twitter', color="orange")
            plt.plot(range(len(soc_ratio)), soc_ratio, marker='o', label='Soc-Twitter-Follows', color="green")
            plt.axhline(y=0.5, linestyle='--')
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Tweet Positivity Ratio: total positive tweets / total tweets")
            plt.title(f"α={a}, β={b}, seeds={seed}, error={round(mae_tweets, 4)}")
            plt.savefig(f"./plots/{a}-{b}-{seed}-tweets.png", bbox_inches='tight', pad_inches=0.1)
            plt.show(block=False)

            # plot nodes
            plt.figure(figsize=(8, 5))
            plt.plot(range(len(higgs_pos_nodes)), higgs_pos_nodes, marker='o', label='Higgs (pos)', color="blue")
            plt.plot(range(len(twitter_pos_nodes)), twitter_pos_nodes, marker='o', label='Twitter (pos)', color="orange")
            plt.plot(range(len(soc_pos_nodes)), soc_pos_nodes, marker='o', label='Soc-Twitter-Follows (pos)', color="green")
            plt.plot(range(len(higgs_neg_nodes)), higgs_neg_nodes, marker='o', label='Higgs (neg)', color="red")
            plt.plot(range(len(twitter_neg_nodes)), twitter_neg_nodes, marker='o', label='Twitter (neg)', color="purple")
            plt.plot(range(len(soc_neg_nodes)), soc_neg_nodes, marker='o', label='Soc-Twitter-Follows (neg)', color="brown")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Total Positive/Negative Nodes")
            plt.title(f"α={a}, β={b}, seeds={seed}")
            plt.savefig(f"./plots/{a}-{b}-{seed}-nodes.png", bbox_inches='tight', pad_inches=0.1)
            plt.show(block=False)

        # plot influencer nodes
        higgs_ratio, higgs_pos_nodes, higgs_neg_nodes = run_model(higgs, a, b, 50, influencer=True)
        twitter_ratio, twitter_pos_nodes, twitter_neg_nodes = run_model(twitter, a, b, 50, influencer=True)
        soc_ratio, soc_pos_nodes, soc_neg_nodes = run_model(soc, a, b, 50, influencer=True)

        time = range(max(len(higgs_ratio), len(twitter_ratio), len(soc_ratio)))

        # plot tweets
        mae_tweets = mae(higgs_ratio, twitter_ratio, soc_ratio)
        plt.figure(figsize=(8, 5))
        plt.fill_between(time, ground_truth, 0, color="red", alpha=0.2)
        plt.fill_between(time, ground_truth, 1, color="green", alpha=0.2)
        plt.plot(range(len(higgs_ratio)), higgs_ratio, marker='o', label='Higgs', color="blue")
        plt.plot(range(len(twitter_ratio)), twitter_ratio, marker='o', label='Twitter', color="orange")
        plt.plot(range(len(soc_ratio)), soc_ratio, marker='o', label='Soc-Twitter-Follows', color="green")
        plt.axhline(y=0.5, linestyle='--')
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Tweet Positivity Ratio: total positive tweets / total tweets")
        plt.title(f"α={a}, β={b}, seeds={seed}, error={round(mae_tweets, 4)}")
        plt.savefig(f"./plots/{a}-{b}-influencer-tweets.png", bbox_inches='tight', pad_inches=0.1)
        plt.show(block=False)

        # plot nodes
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(higgs_pos_nodes)), higgs_pos_nodes, marker='o', label='Higgs (pos)', color="blue")
        plt.plot(range(len(twitter_pos_nodes)), twitter_pos_nodes, marker='o', label='Twitter (pos)', color="orange")
        plt.plot(range(len(soc_pos_nodes)), soc_pos_nodes, marker='o', label='Soc-Twitter-Follows (pos)', color="green")
        plt.plot(range(len(higgs_neg_nodes)), higgs_neg_nodes, marker='o', label='Higgs (neg)', color="red")
        plt.plot(range(len(twitter_neg_nodes)), twitter_neg_nodes, marker='o', label='Twitter (neg)', color="purple")
        plt.plot(range(len(soc_neg_nodes)), soc_neg_nodes, marker='o', label='Soc-Twitter-Follows (neg)', color="brown")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Total Positive/Negative Nodes")
        plt.title(f"α={a}, β={b}, seeds={50}")
        plt.savefig(f"./plots/{a}-{b}-influencer-nodes.png", bbox_inches='tight', pad_inches=0.1)
        plt.show(block=False)
