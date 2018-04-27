import argparse
import csv
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs
from scipy.stats import gamma, spearmanr
import collections

DATAFILE = "2018ausc_qm.csv"

MAX_LL = 1e10

class Data:
    key = 0
    match_number = 1
    red_team = 2
    blue_team = 3
    red_tech_foul_count = 4
    blue_tech_foul_count = 5
    red_foul_points = 6
    blue_foul_points = 7

def extract_teams(s):
    # '{frc6434,frc6024,frc7278}'
    return s.replace("{", "").replace("}", "").split(",")

def read_data(datafile):
    rows = []
    teams = set()
    with open(datafile, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            red = extract_teams(row[Data.red_team])
            blue = extract_teams(row[Data.blue_team])
            teams.update(red)
            teams.update(blue)
            rows += [[red, row[Data.red_tech_foul_count]],
                     [blue, row[Data.blue_tech_foul_count]]]
    team_index = dict([(t, i) for i, t in enumerate(sorted(teams))])
    M = len(team_index)
    X = []
    Y = []
    for row in rows:
        x = [0] * M
        for team in row[0]:
            x[team_index[team]] = 1
        X += [x]
        Y += [int(row[1])]
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, team_index

def generate_synthetic_data(X, N=1, mean=0.1, v=0.01):
    """
    Create synthetic average rates of tech fouls per team by drawing samples from a gamma distribution,
    then create synthetic samples of match fouls using the alliance matrix.
    Uses X N times to create more data for evaluation
    """
    theta = v / mean
    k = mean / theta
    Xout = X
    for i in range(2, N):
        Xout = np.vstack((Xout, X))
    A = np.random.gamma(k, theta, (Xout.shape[1],))
    samples = np.random.poisson(A, size=Xout.shape)
    F = np.sum(Xout * samples, axis=1)
    # also calculate mean of actual samples
    actuals = np.sum(Xout * samples, axis=0) / np.sum(Xout, axis=0)
    return A, Xout, F, actuals

def least_squares(X, Y):
    Aest, resid, rank, s = np.linalg.lstsq(X, Y)
    return Aest

def log_likelihood_fn(A, X, Y):
    if np.any(np.less_equal(A, 1e-6)):
        return MAX_LL
    alliance_sums = np.matmul(X, A)
    if np.any(np.less_equal(alliance_sums, 1e-6)):
        return MAX_LL
    LL = -np.sum(Y * np.log(alliance_sums) - alliance_sums)
    if LL > MAX_LL:
        LL = MAX_LL
    return LL


def log_likelihood_deriv(A, X, Y):
    parts = Y / np.matmul(X, A) - 1
    deriv = np.dot(X.T, parts)
    return -deriv

def log_map_fn(A, X, Y, alpha, beta):
    if np.any(np.less_equal(A, 1e-6)):
        return MAX_LL
    alliance_sums = np.matmul(X, A)
    if np.any(np.less_equal(alliance_sums, 1e-6)):
        return MAX_LL
    LL = -np.sum(Y * np.log(alliance_sums) - alliance_sums) - np.sum(np.log(gamma.pdf(A*beta, alpha)*beta))
    if LL > MAX_LL:
        LL = MAX_LL
    return LL

def log_map_deriv(A, X, Y, alpha, beta):
    parts = Y / np.matmul(X, A) - 1
    deriv = np.dot(X.T, parts) - ((alpha - 1) / A - beta)
    return -deriv

def max_likelihood(X, Y):
    # use synthetic data generator for initial guess
    A0, *_ = generate_synthetic_data(X)
    Aopt = fmin(lambda A: log_likelihood_fn(A, X, Y), A0, xtol=0.001, maxiter=10000)
    return Aopt

def max_likelihood_bfgs(X, Y, A0=None):
    # use synthetic data generator for initial guess
    if A0 is None:
        A0, *_ = generate_synthetic_data(X)
    else:
        A0[A0<0] = 0.001
    Aopt = fmin_bfgs(lambda A: log_likelihood_fn(A, X, Y), A0,
                     fprime=lambda A: log_likelihood_deriv(A, X, Y),
                     gtol=1e-8,
                     maxiter=10000)
    return Aopt

def map_bfgs(X, Y, A0=None):
    # use synthetic data generator for initial guess
    if A0 is None:
        A0, *_ = generate_synthetic_data(X)
    else:
        A0[A0<0] = 0.001
    # estimate priors from data
    m = np.sum(Y) / np.sum(X)
    v = np.var(Y)
    beta = m/v
    alpha = beta * m
    Aopt = fmin_bfgs(lambda A: log_map_fn(A, X, Y, alpha, beta), A0,
                     fprime=lambda A: log_map_deriv(A, X, Y, alpha, beta),
                     gtol=1e-8,
                     maxiter=10000)
    return Aopt

def bootstrap_result(A):
    # summarise bootstrap samples; A is Nbootstrap x M
    print("SHAPE A", np.array(A).shape)
    z = np.percentile(A, [50, 10, 90], axis=0)
    means = np.mean(A, axis=0) # z[0,:]
    print("MEANS", means)
    ranges = z[1:3, :].T
    print("RANGES", ranges)
    return means, ranges

def run_bootstrap(X, Y, Nbootstrap=100):
    selector = np.array(list(range(Y.shape[0])))
    Aest1 = []
    Aest2 = []
    Aest3 = []
    for i in range(Nbootstrap):
        c = np.random.choice(selector, X.shape[0])
        Xb = X[c,:]
        Yb = Y[c]
        Aest1 += [max_likelihood_bfgs(Xb, Yb)]
        Aest2 += [least_squares(Xb, Yb)]
        Aest3 += [map_bfgs(Xb, Yb)]
    m1, r1 = bootstrap_result(Aest1)
    m2, r2 = bootstrap_result(Aest2)
    m3, r3 = bootstrap_result(Aest3)
    return m1, m2, m3, r1, r2, r3

def do_plot(xx, Atest, actuals, Aest1, Ar1, colour, title):
    plt.plot(xx, Atest, 'rx', xx, actuals, 'r.')
    plt.errorbar(xx, Aest1, yerr=np.array([[-1],[1]]) * (Ar1.T-Aest1), fmt=colour + 'o')
    plt.grid()
    plt.xlabel("Team")
    plt.ylabel("Foul Rate")
    plt.title("Foul Rate Estimator, %s" % title)

def run_single_test(X, Y, Ntimes=20, do_report=True):
    Atest, Xtest, Ytest, actuals = generate_synthetic_data(X, Ntimes)
    # Aest1 = max_likelihood(Xtest, Ytest)
    # Aest2 = least_squares(Xtest, Ytest)
    # Aest3 = max_likelihood_bfgs(Xtest, Ytest)
    Aest1, Aest2, Aest3, Ar1, Ar2, Ar3 = run_bootstrap(Xtest, Ytest)
    As = (Aest1, Aest2, Aest3)
    # sum squared error vs underlying means
    eA = [np.sum(np.square(x - Atest)) for x in As]
    # SSE vs actuals
    ea = [np.sum(np.square(x - actuals)) for x in As]
    # spearman coefficient vs underlying means
    sA = [spearmanr(Atest, x).correlation for x in As]
    # spearman coefficient vs actuals
    sa = [spearmanr(actuals, x).correlation for x in As]

    wins = [np.argmin(eA), np.argmin(ea), np.argmax(sA), np.argmax(sa)]
    
    if do_report:
        print("Squared Error:", eA)
        print("Squared Error (v. actuals):", ea)
        print("Spearman:", sA)
        print("Spearman (v. actuals):", sa)
        
        xx = list(range(len(Atest)))
        #plt.plot(xx, Atest, 'r', xx, Aest2, 'g', xx, Aest3, 'b')
        fig0 = plt.figure(0)
        do_plot(xx, Atest, actuals, Aest1, Ar1, 'b', "Maximum Likelihood")
        fig1 = plt.figure(1)
        do_plot(xx, Atest, actuals, Aest2, Ar2, 'g', "Least Squares")
        fig2 = plt.figure(2)
        do_plot(xx, Atest, actuals, Aest3, Ar3, 'y', "MAP")
        plt.show()

    return eA, ea, sA, sa, wins

def summary_plot(eAsumm, title, ylabel):
    xx=["ML", "LS", "MAP"]
    i=[0,1,2]
    plt.errorbar(i, eAsumm[0], yerr=np.array([[-1],[1]]) * (eAsumm[1].T-eAsumm[0]), fmt='ro')
    plt.xticks(i, xx)
    plt.xlabel("Method")
    plt.ylabel(ylabel)
    plt.title(title)

def run_multiple_tests(X, Y, repeats=100, Ntimes=20):
    eAhist = []
    eahist = []
    sAhist = []
    sahist = []
    winshist = [collections.Counter(),collections.Counter(),collections.Counter(),collections.Counter()]

    for i in range(repeats):
        eA, ea, sA, sa, wins = run_single_test(X, Y, Ntimes, do_report=False)
        eAhist += [eA]
        eahist += [ea]
        sAhist += [sA]
        sahist += [sa]
        for wh, w in zip(winshist, wins):
            wh.update({w: 1})

    print("SHAPE eA", np.array(eAhist).shape)
    eAsumm = bootstrap_result(eAhist)
    easumm = bootstrap_result(eahist)
    sAsumm = bootstrap_result(sAhist)
    sasumm = bootstrap_result(sahist)

    print(eAsumm, easumm, sAsumm, sasumm, winshist)

    plt.subplot(221)
    summary_plot(eAsumm, "SSE of Estimates of Underlying Mean", "SSE")
    plt.subplot(222)
    summary_plot(easumm, "SSE of Estimates of Observed Mean", "SSE")
    plt.subplot(223)
    summary_plot(sAsumm, "Spearman Rank Correlation of Estimates of Underlying Mean", "Spearman Rank Correlation")
    plt.subplot(224)
    summary_plot(sasumm, "Spearman Rank Correlation of Estimates of Observed Mean", "Spearman Rank Correlation")
    plt.show()
    
def run_real_data(X, Y, team_index, Nbootstrap=1000):
    A1, A2, A3, Ar1, Ar2, Ar3 = run_bootstrap(X, Y, Nbootstrap)
    index_to_team = []
    for team, i in sorted(team_index.items()):
        print("{0:3d} {1:5s} {2:4.2f} {3:4.2f} {4:4.2f}".format(i, team, A1[i], A2[i], A3[i]))
        index_to_team += [team]
        
    for i in np.argsort(-A3):
        print("{0:3d} {1:5s} {2:4.2f} {3:4.2f} {4:4.2f}".format(i, index_to_team[i], A1[i], A2[i], A3[i]))
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        type=str,
        help="data file to load",
        default=DATAFILE)

    parser.add_argument(
        "--one-test",
        "-t",
        action="store_true",
        help="Run one test and display results",
        default=False)

    parser.add_argument(
        "--long-test",
        "-l",
        action="store_true",
        help="Run multiple tests and summarise results",
        default=False)

    parser.add_argument(
        "--competitions",
        "-c",
        type=int,
        help="Equivalent number of competitions to run test data on",
        default=1)

    parser.add_argument(
        "--repeats",
        "-i",
        type=int,
        help="Number of repeats to run in long test",
        default=100)
    
    parser.add_argument(
        "--real",
        "-r",
        action="store_true",
        help="Use the real data and summarise results",
        default=False)
    
    args = parser.parse_args()

    X, Y, team_index = read_data(args.data_file)

    if args.one_test:
        run_single_test(X, Y, args.competitions)

    if args.long_test:
        run_multiple_tests(X, Y, repeats=args.repeats, Ntimes=args.competitions)
        
    if args.real:
        run_real_data(X, Y, team_index)
        
