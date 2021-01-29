import argparse
import numpy as np
import sys
import os
from train_GCE import train_GCE


def run_algorithm_1(args):
    dataset = args.dataset
    classes_used = args.classes_used
    output_dir = args.output_dir

    # Starting values of training parameters
    K = 0
    L = args.L
    lam = 0.01
    L_step = args.L_step
    lam_step = args.lam_step
    train_steps = args.train_steps
    
    # criteria step 1 (in percentual improvement between consecutive runs)
    criteria = args.step1_crit
    # criteria step 2
    C_crit = args.step2_C_crit
    D_crit = args.step2_D_crit
    
    ## STEP 1
    D_optimal, L_optimal, vary_L_results = step_1(dataset, classes_used, K, L,
                                                  lam, train_steps, L_step,
                                                  criteria=criteria)
    ## STEP 2/3
    vary_K_L_lambda_results = step_2(dataset, classes_used, K, L_optimal, lam,
                                     train_steps, lam_step, D_optimal,
                                     C_crit=C_crit, D_crit=D_crit)
    
    # print results
    print_results(vary_L_results, vary_K_L_lambda_results)
    # save results
    save_results(vary_L_results, vary_K_L_lambda_results, dataset, classes_used,
                 output_dir)


def step_1(dataset, classes_used, K, L, lam, train_steps, L_step, criteria=1):
    # init variables
    D = 999
    D_rel_improvement = 999
    vary_L_results = [['L', 'D_results']]
    
    # stop when improvement is <1 (so negative improvement is also bad)
    while D_rel_improvement > criteria:
        print('\nTraining with K={}, L={}, lambda={}'.format(K, L, lam))
        train_results = train_GCE(f"base_{dataset}_{classes_used}_classifier",
                                  K, L, lam=lam, train_steps=train_steps,
                                  retrain=True)

        # retrieve average of last 500 training steps to compare with previous run
        D_new = np.mean(train_results['loss_nll'][-500:])
        
        # relative improvement of distance D
        D_rel_improvement = -1 * ((D_new - D) / D * 100)
        print('Using L={}, the relative improvement of D_new: {:.2f}%'.format(L,
                                                                              D_rel_improvement))
        
        # save results
        vary_L_results.append([L, D_new])
        
        # init for next cycle
        L += L_step
        D = D_new
    
    # need L from before last cycle
    L_optimal = L - L_step
    print('Optimal L={}'.format(L_optimal))
    
    # return optimal D and L before plateau is reached
    return D, L_optimal, vary_L_results


def step_2(dataset, classes_used, K, L, lam, train_steps,
           lam_step, D_optimal, C_crit=1, D_crit=1):
    # init variables
    C_rel_improvement = 999
    C = -0.000000001
    D_rel_diff = 999
    D_rel_diff_old = 999
    lam_use = 0
    vary_K_L_lambda_results = [['K', 'L', 'lambda', 'C', 'D', 'total_loss']]

    # change K,L,lambda until C plateaus
    while C_rel_improvement > C_crit:  # % improvement on distance
        K += 1
        L -= 1
        print("\nNow training with K={} and L={}".format(K, L))
        
        while D_rel_diff > D_crit:
            lam_use = round(lam_use + lam_step, 2)
            print("\nTraining with lambda={}".format(lam_use))

            train_results = train_GCE(f"base_{dataset}_{classes_used}_classifier",
                                      K, L, lam=lam, train_steps=train_steps, retrain=True)
            
            # calculate relative difference of distance D
            D_new = np.mean(train_results['loss_nll'][-500:])
            D_rel_diff = ((D_new - D_optimal) / D_optimal * 100)
            print('D_new={}, D={}'.format(D_new, D_optimal))
            print('D is still {:.2f}% worse than D_optimal'.format(D_rel_diff))
            
            # if the D_new is becoming worse, stop iterating and use previous lambda
            if D_rel_diff > D_rel_diff_old:
                lam_use -= lam_step
                break
            
            D_rel_diff_old = D_rel_diff
        
        print("Optimal lambda={}".format(lam_use))
        
        # if C approaches optimal C, save causal effect
        C_new = np.mean(train_results['loss_ce'][-500:])
        C_rel_improvement = (C_new - C) / C * 100
        print('C_new={}, C={}'.format(C_new, C))
        print('Relative improvement of causal effect: {:.2f}%'.format(
            C_rel_improvement))
        
        # save all variables per step
        total_loss = np.mean(train_results['loss'][-500:])
        vary_K_L_lambda_results.append(
            [K, L, lam_use, C_new, D_new, total_loss])
        
        # init for next cycle
        C = C_new
        D_rel_diff = 999
        D_rel_diff_old = 999
        lam_use = 0

        # stop execution when L reached 0
        if L == 0:
            print('C did not reach plateau, and L is 0. \n Set C_crit higher, or do more training steps')
            return vary_K_L_lambda_results
            
    return vary_K_L_lambda_results


def print_results(vary_L_results, vary_K_L_lambda_results):
    # print results
    print(vary_L_results)
    print(vary_K_L_lambda_results)
    
    # retrieve optimal parameters before plateau of C
    K_optimal = vary_K_L_lambda_results[-2][0]
    L_optimal = vary_K_L_lambda_results[-2][1]
    lam_optimal = vary_K_L_lambda_results[-2][2]
    
    print("Optimal K={}, L={}, lambda={}".format(K_optimal, L_optimal,
                                                 lam_optimal))


def save_results(vary_L_results, vary_K_L_lambda_results, dataset, classes_used,
                 output_dir):
    save_dir = output_dir + '{}_{}_'.format(dataset, str(classes_used))

    os.makedirs(output_dir, exist_ok=True)

    np.save(save_dir + 'algorithm1_step_1_results.npy', vary_L_results)
    np.save(save_dir + 'algorithm1_step_2_results.npy', vary_K_L_lambda_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='fmnist',
                        help="Specification of used dataset")
    parser.add_argument("--classes_used", type=str, default='034',
                        help="Specification of classes used from dataset")
    parser.add_argument("--L", type=int, default=1,
                        help="Start value for L in step 1")
    parser.add_argument("--train_steps", type=int, default=3000,
                        help="Specification of training steps for GCE.")
    parser.add_argument("--L_step", type=int, default=1,
                        help="Specification of incremental stepsize of L in step 1")
    parser.add_argument("--lam_step", type=float, default=0.01,
                        help="Specification of incremental stepsize of lambda in step 2")
    parser.add_argument("--step1_crit", type=int, default=1,
                        help="Criteria for D plateauing in step 1, "
                             "stops when relative change is lower than 'step1_crit'%")
    parser.add_argument("--step2_C_crit", type=int, default=1,
                        help="Criteria for C plateauing in step 2,"
                             "stops when relative change is lower than 'step2_C_crit'%")
    parser.add_argument("--step2_D_crit", type=int, default=5,
                        help="Criteria for D approaching optimal D from step 1,"
                             "stops when relative difference is lower than 'step2_D_crit'%")
    parser.add_argument("--output_dir", type=str, default='reports/parameters/algorithm1/',
                        help="Output directory")
    args = parser.parse_args()

    # run the algorithm
    run_algorithm_1(args)