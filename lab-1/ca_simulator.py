# CS 420/CS 527 Lab 1: "Edge of Chaos" with Cellular Automata
# Adapted from Java program written by student in CS 420 in 2007
# Original Source found: http://web.eecs.utk.edu/~bmaclenn/Classes/420-527-S18/projects/project1/ca.html
# Adapted by Catherine Schuman
# January 2022

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from itertools import product


class CASimulator:
    def __init__(self, num_experiments, seed, fn="MasterExperiment.csv", exp_dir="experiments"):
        self.num_experiments = num_experiments
        np.random.seed(seed)
        self.seed = seed
        self.num_states = 5
        self.radius = 1
        self.neighborhood = 2*self.radius+1
        self.fn = fn
        self.exp_dir = exp_dir
        self.dim = 200
        self.timesteps = 1000
        self.rule_table, self.board = None, None

        # palette = ["black", "blue", "green", "yellow", "orange"]
        self.palette = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 0], [255, 165, 0]])

    def simulate(self, debug=False):
        f = open(self.fn, "w")

        f.write("Your Name:,Clark Hathaway\n")
        f.write("Wrap:,true\n")
        f.write("K(states):," + str(self.num_states) + "\n")
        f.write("Radius:," + str(self.radius) + "\n")
        f.write("Quiescence:,true\n")    
        f.write("random seed:," + str(self.seed) + "\n")
        
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        for experiment in range(self.num_experiments):
            # Initialize the board for this experiment
            self.board = np.zeros((self.timesteps, self.dim))
            self.board[0, :] = np.random.randint(0, self.num_states, self.dim)

            # Randomly initialize the rule
            self.rule_table = [0 if n == 0 else np.random.randint(1, self.num_states) for n in range(13)]
            to_be_decimated = [n for n, _ in enumerate(self.rule_table) if 0 < n]

            rule_string = ''.join((str(x) for x in self.rule_table))
            
            # Add in writing to experiment file
            f.write("\n\n")
            f.write("Experiment #:," + str(experiment)+"\n")
            f.write("Rule:," + rule_string+"\n")
            f.write("Step,Entry Zeroed,Class,Lambda,Lambda_t,H,H_t,Zeta,Observations\n")
            
            index_to_0 = "-"
            for z, _ in enumerate(self.rule_table):
                tot_lam = self.calculate_totalistic_lambda()
                lam = self.calculate_lambda()
                tot_entropy = self.calculate_totalistic_entropy()
                entropy = self.calculate_entropy()

                entry_zeroed = index_to_0

                f.write(','.join((
                    str(z), str(entry_zeroed), "," + str(lam), str(tot_lam), str(entropy), str(tot_entropy), ",\n"
                )))
                
                # Randomly select one to be decimated and remove it
                if 0 < len(to_be_decimated):
                    index = np.random.randint(0, len(to_be_decimated))
                    index_to_0 = to_be_decimated.pop(index)

                # Step through time updating the board
                self.board[1:, :] = [
                    [
                        int(self.rule_table[int(self.calculate_my_sum(r, c))])
                        for c in range(self.dim)
                    ]
                    for r in range(self.timesteps - 1)
                ]

                # Create the associated figure 
                fig = plt.figure()
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off() 
                fig.add_axes(ax)
                ax.imshow(self.palette[self.board.astype(int)]) 
                img_fn = self.exp_dir + "/experiment_" + str(experiment) + "_step_" + str(z) + ".pdf"
                plt.savefig(img_fn, dpi=800, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                # Zero out one of the rule tables
                self.rule_table[index_to_0] = 0

                if debug:
                    print(f"experiment {experiment:02d}: step {z:02d} complete")
        f.close() 

    def calculate_totalistic_lambda(self):
        num0 = 0
        for x in range(len(self.rule_table)):
            if self.rule_table[x] == 0:
                num0 += 1
            
        return 1.0-(float(num0)/float(len(self.rule_table)))

    def calculate_lambda(self):
        d = [1, 3, 6, 10, 15, 18, 19, 18, 15, 10, 6, 3, 1]
        num0 = 0
        for x in range(len(self.rule_table)):
            new_state = self.rule_table[x]
            if new_state == 0:
                num0 += d[x]
        
        return 1.0 - (float(num0) / np.power(self.num_states, self.neighborhood))

    def calculate_totalistic_entropy(self):
        state_occurrence = [0]*self.num_states
        for x, _ in enumerate(self.rule_table):
            state_occurrence[self.rule_table[x]] += 1

        result = 0
        for x in range(self.num_states):
            ps = float(state_occurrence[x])/float(len(self.rule_table))
            if ps != 0:
                result += (ps*np.log2(ps))

        return -1*result

    def calculate_entropy(self):
        d = [1, 3, 6, 10, 15, 18, 19, 18, 15, 10, 6, 3, 1]
        state_occurrence = [0]*self.num_states
        result = 0
        for x in range(len(self.rule_table)):
            new_state = self.rule_table[x]
            state_occurrence[new_state] += d[x]

        for x in range(self.num_states):
            ps = float(state_occurrence[x])/np.power(self.num_states, self.neighborhood)
            if ps != 0:
                result += ps*np.log2(ps)

        return -result

    def calculate_my_sum(self, r, c):
        row = self.board[r]

        return sum(
            row[index + self.dim * (index < 0)] for index in (c - x - 1 for x in range(self.radius))
        ) + sum(
            row[index - self.dim * (self.dim <= index)] for index in (c + x for x in range(self.radius + 1))
        )


def main():
    parser = argparse.ArgumentParser(description="Cellular Automata -- CS 420/527 Lab 1")
    parser.add_argument("--experiments", "-e", required=True, type=int, help="Number of experiments")
    parser.add_argument("--seed", required=False, type=int, default=-123, help="RNG seed (default is the time")
    parser.add_argument("--exp_dir", type=str, default="experiments", help="Directory to put experiment files.")
    parser.add_argument("--master_file", type=str, default="MasterExperiment.csv", help="Master file name")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.seed == -123:
        seed = int(time.time())
    else:
        seed = args.seed

    cas = CASimulator(args.experiments, seed, args.master_file, args.exp_dir)
    cas.simulate(debug=args.debug)


if __name__ == '__main__':
    main()
