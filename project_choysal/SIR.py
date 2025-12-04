import random
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import mmread
import csv
import os

S = 0
I_POS = 1
I_NEG = 2
R_POS = 3
R_NEG = 4

def dual_sir_single(
    G,
    beta_pos,
    beta_neg,
    gamma_pos,
    gamma_neg,
    seeds_pos,
    seeds_neg,
    max_steps=15,
    directed=True,
    rng=None,
):
    if rng is None:
        rng = random.Random()

    state = {u: S for u in G.nodes()}
    ever_pos = set()
    ever_neg = set()

    # initialize seeds
    for u in seeds_pos:
        state[u] = I_POS
        ever_pos.add(u)
    for u in seeds_neg:
        state[u] = I_NEG
        ever_neg.add(u)

    def neigh(u):
        if directed and isinstance(G, nx.DiGraph):
            return G.predecessors(u)
        return G.neighbors(u)

    hist_ratio = []
    hist_inf_pos = []
    hist_inf_neg = []

    for _ in range(max_steps):
        vals = list(state.values())
        cIpos = vals.count(I_POS)
        cIneg = vals.count(I_NEG)
        cRpos = vals.count(R_POS)
        cRneg = vals.count(R_NEG)

        pos_total = cIpos + cRpos
        neg_total = cIneg + cRneg

        hist_inf_pos.append(cIpos)
        hist_inf_neg.append(cIneg)
        hist_ratio.append(pos_total / (pos_total + neg_total) if pos_total + neg_total > 0 else 0)

        # stop if no infection left
        if cIpos == 0 and cIneg == 0:
            break

        new_pos, new_neg = set(), set()
        rec_pos, rec_neg = set(), set()

        inf_pos = [u for u in state if state[u] == I_POS]
        inf_neg = [u for u in state if state[u] == I_NEG]

        # infections
        for u in inf_pos:
            for v in neigh(u):
                if state[v] == S and rng.random() < beta_pos:
                    new_pos.add(v)

        for u in inf_neg:
            for v in neigh(u):
                if state[v] == S and rng.random() < beta_neg:
                    new_neg.add(v)

        # resolve conflicts
        conflict = new_pos & new_neg
        for v in conflict:
            if rng.random() < 0.5:
                new_neg.remove(v)
            else:
                new_pos.remove(v)

        # recoveries
        for u in inf_pos:
            if rng.random() < gamma_pos:
                rec_pos.add(u)
        for u in inf_neg:
            if rng.random() < gamma_neg:
                rec_neg.add(u)

        # apply new infections
        for v in new_pos:
            if state[v] == S:
                state[v] = I_POS
                ever_pos.add(v)

        for v in new_neg:
            if state[v] == S:
                state[v] = I_NEG
                ever_neg.add(v)

        # apply recover
        for v in rec_pos:
            if state[v] == I_POS:
                state[v] = R_POS
        for v in rec_neg:
            if state[v] == I_NEG:
                state[v] = R_NEG

    return {
        "ever_pos": len(ever_pos),
        "ever_neg": len(ever_neg),
        "history": {
            "pos_ratio": hist_ratio,
            "pos_infected": hist_inf_pos,
            "neg_infected": hist_inf_neg,
        }
    }

def load_graph(path, directed=True):
    if path.endswith(".mtx"):
        adj = mmread(path)
        G = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)
        return nx.relabel_nodes(G, int)
    return nx.read_edgelist(path, create_using=nx.DiGraph if directed else nx.Graph)


def influencer_seeds(G, k=100):
    deg_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in deg_sorted[:k]]

def average_series(runs, key):
    max_len = max(len(r[key]) for r in runs)

    padded = []
    for r in runs:
        arr = r[key].copy()
        while len(arr) < max_len:
            arr.append(arr[-1])
        padded.append(arr)

    return [
        sum(run[t] for run in padded) / len(padded)
        for t in range(max_len)
    ]

def plot_total_infected(histories, title, filename):
    plt.figure(figsize=(10,6))
    for name, hist in histories.items():
        plt.plot(hist["pos_infected"], "-o", label=f"{name} (pos)")
        plt.plot(hist["neg_infected"], "-o", label=f"{name} (neg)")
    plt.xlabel("Timestep")
    plt.ylabel("Total Positive/Negative Nodes")
    plt.title(title)
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_pos_ratio(histories, title, filename):
    plt.figure(figsize=(10,6))
    for name, hist in histories.items():
        plt.plot(hist["pos_ratio"], "-o", label=name)
    plt.axhline(0.5, linestyle="--", color="blue")
    plt.ylim(0,1)
    plt.xlabel("Timestep")
    plt.ylabel("Positivity Ratio")
    plt.title(title)
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

if __name__ == "__main__":

    rng = random.Random(1)
    RUNS_PER_SETTING = 15
    os.makedirs("plots", exist_ok=True)

    # load networks
    networks = {
        "Higgs": load_graph("higgs-social_network.edgelist"),
        "Twitter": load_graph("twitter_combined.txt"),
        "Soc-Twitter-Follows": load_graph("soc-twitter-follows.mtx"),
    }

    seed_modes = ["2500", "7500", "influencer"]
    beta_levels = [0.1, 0.02]
    gamma_levels = [0.2, 0.5]

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["network","seed_mode","beta","gamma","pos_total","neg_total","final_ratio"])

    for seed_mode in seed_modes:
        for beta in beta_levels:
            for gamma in gamma_levels:

                print(f"\nRUNNING {seed_mode} β={beta} γ={gamma}")

                avg_histories = {}
                run_final_ratios = []

                for name, G in networks.items():
                    runs = []

                    for _ in range(RUNS_PER_SETTING):
                        nodes = list(G.nodes())

                        if seed_mode == "2500":
                            pos = rng.sample(nodes, 2500)
                            neg = rng.sample(list(set(nodes) - set(pos)), 2500)
                        elif seed_mode == "7500":
                            pos = rng.sample(nodes, 7500)
                            neg = rng.sample(list(set(nodes) - set(pos)), 7500)
                        else:
                            top = influencer_seeds(G, 100)
                            pos = rng.sample(top, 50)
                            neg = rng.sample(list(set(top) - set(pos)), 50)

                        R = dual_sir_single(G, beta, beta, gamma, gamma, pos, neg, rng=rng)
                        runs.append(R["history"])

                        pos_n = R["ever_pos"]
                        neg_n = R["ever_neg"]
                        ratio = pos_n / (pos_n + neg_n) if pos_n + neg_n > 0 else 0
                        run_final_ratios.append(ratio)

                    avg_pos_infected = average_series(runs,"pos_infected")
                    avg_neg_infected = average_series(runs,"neg_infected")
                    avg_pos_ratio = average_series(runs,"pos_ratio")

                    with open("results.csv", "a", newline="") as f:
                        csv.writer(f).writerow([name,seed_mode,beta,gamma,avg_pos_infected[-1],avg_neg_infected[-1],avg_pos_ratio[-1]])

                    avg_histories[name] = {
                        "pos_infected": avg_pos_infected,
                        "neg_infected": avg_neg_infected,
                        "pos_ratio": avg_pos_ratio,
                    }

                network_avg_ratios = [
                    avg_histories[name]["pos_ratio"][-1]
                    for name in networks.keys()
                ]
                avg_error = sum(abs(r - 0.5) for r in network_avg_ratios) / len(network_avg_ratios)


                title = f"β={beta}, γ={gamma}, seeds={seed_mode}, error={avg_error:.4f}"

                plot_total_infected(
                    avg_histories,
                    title,
                    f"plots/TOTAL_{seed_mode}_b{beta}_g{gamma}.png"
                )

                plot_pos_ratio(
                    avg_histories,
                    title,
                    f"plots/RATIO_{seed_mode}_b{beta}_g{gamma}.png"
                )
