import os
import sys
import six

sys.modules['sklearn.externals.six'] = six

import mlrose_hiive as mlrose
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


class Problem(Enum):
    KNAPSACK = 1
    NQUEEN = 2
    FOUR_PEAKS = 3
    MAXKCOLOR = 4


class Optimisation(Enum):
    RHC = 1
    SA = 2
    GA = 3
    MIMIC = 4


__MAX_ATTEMPTS = {Problem.KNAPSACK: 100,
                  Problem.NQUEEN: 100,
                  Problem.FOUR_PEAKS: 100}

__POP_SIZE = {Problem.KNAPSACK: 500,
              Problem.NQUEEN: 500,
              Problem.FOUR_PEAKS: 300}

__MUTATION_PROBABILITY = 0.1
__KEEP_PCT = 0.175
__MAX_ITERS = 1000


def set_mut_prob(prob):
    global __MUTATION_PROBABILITY
    __MUTATION_PROBABILITY = prob


def set_keep_pct(pct):
    global __KEEP_PCT
    __KEEP_PCT = pct


def set_max_iters(iters):
    global __MAX_ITERS
    __MAX_ITERS = iters


def set_max_attempts(p_type, max_attempts):
    global __MAX_ATTEMPTS
    __MAX_ATTEMPTS[p_type] = max_attempts


def set_pop_size(p_type, pop_size):
    global __POP_SIZE
    __POP_SIZE[p_type] = pop_size


def get_fitness_function(problem_type):
    fitness, problem = None, None

    if problem_type == Problem.KNAPSACK:
        fitness = mlrose.Knapsack(weights=[70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120],
                                  values=[1.35, 1.39, 1.49, 1.50, 1.56, 1.63, 1.73, 1.84, 1.92, 2.01, 2.10, 2.14, 2.21,
                                          2.29, 2.40],
                                  max_weight_pct=0.52)
        problem = mlrose.DiscreteOpt(length=15, fitness_fn=fitness, maximize=True, max_val=2)

    elif problem_type == Problem.NQUEEN:
        fitness = mlrose.Queens()
        problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=False, max_val=8)

    elif problem_type == Problem.FOUR_PEAKS:
        fitness = mlrose.FourPeaks(t_pct=0.15)
        problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)

    return fitness, problem


def get_random_optimisation(problem, optimisation, problem_type, seed=10, state_fitness_callback=None):
    best_state, best_fitness, fitness_curve = None, None, None
    process_time = []

    if optimisation == Optimisation.RHC:
        start = time.time()
        best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem,
                                                                           max_attempts=__MAX_ATTEMPTS[problem_type],
                                                                           max_iters=__MAX_ITERS,
                                                                           curve=True, random_state=seed,
                                                                           state_fitness_callback=state_fitness_callback,
                                                                           callback_user_info=['rhc'])
        process_time.append(time.time() - start)

    elif optimisation == Optimisation.SA:
        start = time.time()
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule=mlrose.ExpDecay(exp_const=0.1),
                                                                             max_attempts=__MAX_ATTEMPTS[problem_type],
                                                                             max_iters=__MAX_ITERS,
                                                                             curve=True, random_state=seed,
                                                                             state_fitness_callback=state_fitness_callback,
                                                                             callback_user_info=['sa'])
        process_time.append(time.time() - start)

    elif optimisation == Optimisation.GA:
        start = time.time()
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=__POP_SIZE[problem_type],
                                                                     mutation_prob=__MUTATION_PROBABILITY,
                                                                     max_attempts=__MAX_ATTEMPTS[problem_type],
                                                                     max_iters=__MAX_ITERS,
                                                                     curve=True, random_state=seed,
                                                                     state_fitness_callback=state_fitness_callback,
                                                                     callback_user_info=['ga'])
        process_time.append(time.time() - start)

    elif optimisation == Optimisation.MIMIC:
        start = time.time()
        best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=__POP_SIZE[problem_type],
                                                               keep_pct=__KEEP_PCT,
                                                               max_iters=__MAX_ITERS,
                                                               max_attempts=__MAX_ATTEMPTS[problem_type],
                                                               curve=True, random_state=seed,
                                                               state_fitness_callback=state_fitness_callback,
                                                               callback_user_info=['mimic'])
        process_time.append(time.time() - start)

    return best_state, best_fitness, np.array(fitness_curve), process_time


def plot_comparison_curves(problems, optimisations):
    i = 0
    curves = []
    times = []
    output = {'optimisation': [], 'problem': [], 'best_state': [], 'best_score': [], 'time': [], 'func_evals': []}
    print("\n Progress: ")
    for prob_type in problems:
        fit_fn, prob_obj = get_fitness_function(prob_type)
        curves.append([])
        times.append([])
        for opt in optimisations:
            state, score, curve, process_time = get_random_optimisation(prob_obj, opt, prob_type)
            output['best_score'].append(np.round(score, 4))
            output['best_state'].append(state)
            output['optimisation'].append(opt)
            output['problem'].append(prob_type)
            curves[i].append(curve[:, 0])
            output['time'].append(process_time)
            if len(curve) < __MAX_ATTEMPTS[prob_type] or curve[-1][0] > curve[-(__MAX_ATTEMPTS[prob_type])][0]:
                output['func_evals'].append(curve[-1][1])
            else:
                output['func_evals'].append(curve[-(__MAX_ATTEMPTS[prob_type])][1])
            print(".", end="")
        i += 1

    df = pd.DataFrame(data=output)
    print()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
    df.to_csv("output.csv")

    curves = np.array(curves)

    for i in range(len(problems)):
        plt.figure(i + 1)
        plt.plot(curves[i][0], label=str(optimisations[0])[13:], color='red')
        plt.plot(curves[i][1], label=str(optimisations[1])[13:], color='blue')
        plt.plot(curves[i][2], label=str(optimisations[2])[13:], color='green')
        plt.plot(curves[i][3], label=str(optimisations[3])[13:], color='orange')
        plt.legend()
        plt.xlabel("iterations")
        plt.ylabel("fitness score")
        plt.title(str(problems[i])[8:])
        plt.savefig("FITNESS_CURVES" + "/fitness_" + str(problems[i])[8:] + ".png")


def plot_sa_graph(prob_type):
    try:
        os.mkdir("./SA")
    except FileExistsError:
        pass
    except OSError as error:
        print("Error creating directory results.")

    print('\n plot_sa_graph - Progress: ', end="")

    fitness, problem = get_fitness_function(prob_type)
    process_time = []
    fitness_score = []
    for j in range(3):
        process_time.append([])
        fitness_score.append([])
        if j != 2:
            decay_const = [0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.9, 0.99]
        else:
            decay_const = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99]
        for i in decay_const:
            print('.', end="")
            if j == 0:
                decay = mlrose.GeomDecay(decay=i)
            elif j == 1:
                decay = mlrose.ExpDecay(exp_const=i)
            else:
                decay = mlrose.ArithDecay(decay=i)
            start = time.time()
            best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem,
                                                                                 schedule=decay,
                                                                                 max_attempts=1000, max_iters=100,
                                                                                 random_state=10)
            fitness_score[j].append(best_fitness)
            process_time[j].append(time.time() - start)

    plt.figure(100)
    plt.plot([0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.9, 0.99], fitness_score[0], 'r', label='GeomDecay')
    plt.plot([0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.9, 0.99],
                fitness_score[1], 'b', label='ExpDecay')
    plt.plot([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99],
                fitness_score[2], 'g', label='ArithDecay')
    plt.xlim(0.0001, 0.2)
    plt.legend()
    plt.xlabel("decay constant")
    plt.ylabel("fitness score")
    plt.title('Simulated Annealing')
    plt.savefig("SA/" + "Fitness.png")

    plt.figure(101)
    plt.plot([0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.9, 0.99], process_time[0], 'r', label='GeomDecay')
    plt.plot([0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.9, 0.99],
             process_time[1], 'b', label='ExpDecay')
    plt.plot([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99],
             process_time[2], 'g', label='ArithDecay')
    plt.xlim(0.0001, 0.2)
    plt.legend()
    plt.xlabel("decay constant")
    plt.ylabel("process time")
    plt.title('Simulated Annealing')
    plt.savefig("SA/" + "Time.png")


def plot_ga_graph(prob_type):
    try:
        os.mkdir("./GA")
    except FileExistsError:
        pass
    except OSError as error:
        print("Error creating directory results.")

    print('\n plot_ga_graph - Progress: ', end="")

    fitness, problem = get_fitness_function(prob_type)
    process_time = []
    fitness_score = []
    population = [100, 200, 300, 500, 700, 1000, 1500]
    for i in population:
        print(".", end="")
        start = time.time()
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=i,
                                                                     mutation_prob=__MUTATION_PROBABILITY,
                                                                     max_attempts=__MAX_ATTEMPTS[prob_type],
                                                                     max_iters=__MAX_ITERS, random_state=10)
        fitness_score.append(best_fitness)
        process_time.append(time.time() - start)

    plt.figure(300)
    plt.plot(fitness_score)
    plt.xticks(np.arange(len(population)), [str(p) for p in population])
    plt.xlabel("pop_size constant")
    plt.ylabel("fitness score")
    plt.title('Genetic Algorithm')
    plt.savefig("GA/" + "Fitness.png")

    plt.figure(301)
    plt.plot(process_time)
    plt.xticks(np.arange(len(population)), [str(p) for p in population])
    plt.xlabel("pop_size constant")
    plt.ylabel("time")
    plt.title('GA')
    plt.savefig("GA/" + "Time.png")


def plot_mimic_graph(prob_type, flag=0):
    try:
        os.mkdir("./MIMIC")
    except FileExistsError:
        pass
    except OSError as error:
        print("Error creating directory results.")

    print('\n plot_mimic_graph - Progress: ', end="")

    fitness, problem = get_fitness_function(prob_type)
    process_time = []
    fitness_score = []
    if flag == 0:
        population = [100, 200, 300, 500, 700, 1000, 1500]
    else:
        population = [0.1, 0.15, 0.175, 0.2, 0.25, 0.3, 0.5]
    for i in population:
        print(".", end="")
        start = time.time()
        if flag == 0:
            best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=i, keep_pct=__KEEP_PCT,
                                                                   max_iters=__MAX_ITERS,
                                                                   max_attempts=__MAX_ATTEMPTS[prob_type],
                                                                   random_state=10)
        else:
            best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=__POP_SIZE[prob_type], keep_pct=i,
                                                                   max_iters=__MAX_ITERS,
                                                                   max_attempts=__MAX_ATTEMPTS[prob_type],
                                                                   random_state=10)
        fitness_score.append(best_fitness)
        process_time.append(time.time() - start)

    plt.figure(200 + flag)
    plt.plot(fitness_score)
    plt.xticks(np.arange(len(population)), [str(p) for p in population])
    plt.xlabel("pop_size constant" if flag == 0 else "keep_pct constant")
    plt.ylabel("fitness score")
    plt.title('MIMIC')
    plt.savefig("MIMIC/" + str(flag) + "Fitness.png")

    plt.figure(210 + flag)
    plt.plot(process_time)
    plt.xticks(np.arange(len(population)), [str(p) for p in population])
    plt.xlabel("pop_size constant" if flag == 0 else "keep_pct constant")
    plt.ylabel("time")
    plt.title('MIMIC')
    plt.savefig("MIMIC/" + str(flag) + "Time.png")


def plot_func_eval(optimisations):
    func_evals = []
    print("\n plot_func_eval: ", end="")

    prob_type = Problem.FOUR_PEAKS
    input_size = [10, 20, 40, 60, 80, 100]
    k = 0
    for i in input_size:
        fitness = mlrose.FourPeaks(t_pct=0.15)
        problem = mlrose.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True, max_val=2)

        func_evals.append([])
        for opt in optimisations:
            state, score, curve, process_time = get_random_optimisation(problem, opt, prob_type)
            if len(curve) < __MAX_ATTEMPTS[prob_type] or curve[-1][0] > curve[-(__MAX_ATTEMPTS[prob_type])][0]:
                func_evals[k].append(curve[-1][1])
            else:
                func_evals[k].append(curve[-(__MAX_ATTEMPTS[prob_type])][1])
            print(".", end="")
        k += 1

    func_evals = np.array(func_evals)

    plt.figure(400)
    plt.plot(func_evals[:, 0], label=str(optimisations[0])[13:], color='red')
    plt.plot(func_evals[:, 1], label=str(optimisations[1])[13:], color='blue')
    plt.plot(func_evals[:, 2], label=str(optimisations[2])[13:], color='green')
    plt.plot(func_evals[:, 3], label=str(optimisations[3])[13:], color='orange')
    plt.legend()
    plt.xticks(np.arange(len(input_size)), [str(z) for z in input_size])
    plt.xlabel("input size")
    plt.ylabel("function evalutations")
    plt.title(str(prob_type)[8:])
    plt.savefig("FITNESS_CURVES" + "/fneval_vs_ipsize_" + str(prob_type)[8:] + ".png")


if __name__ == '__main__':
    try:
        os.mkdir("./FITNESS_CURVES")
    except FileExistsError:
        pass
    except OSError as error:
        print("Error creating directory results.")

    plot_individuals = False
    plot_comparison = True

    problems = [Problem.KNAPSACK, Problem.NQUEEN, Problem.FOUR_PEAKS]
    optimisations = [Optimisation.RHC, Optimisation.SA, Optimisation.GA, Optimisation.MIMIC]

    # Individual Algorithm analysis
    if plot_individuals:
        plot_sa_graph(Problem.NQUEEN)
        plot_mimic_graph(Problem.KNAPSACK, flag=0)
        plot_mimic_graph(Problem.KNAPSACK, flag=1)
        plot_ga_graph(Problem.FOUR_PEAKS)
        plot_func_eval(optimisations)

    # Comparison between different problems and RO
    np.random.seed(1000)
    if plot_comparison:
        plot_comparison_curves(problems, optimisations)
