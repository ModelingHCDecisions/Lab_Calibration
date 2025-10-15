import scipy.stats as stats

import calib_sets as sets
from SurvivalModelClasses import Cohort
from deampy.calibration import CalibrationRandomSampling, CalibrationMCMCSampling


def log_likelihood_func(thetas, seed):
    """Compute the log-likelihood of observed data given theta."""

    cohort = Cohort(id=seed, pop_size=sets.SIM_POP_SIZE, mortality_prob=thetas[0])
    cohort.simulate(n_time_steps=sets.TIME_STEPS, seed=seed)  # Simulate the cohort

    # get the average survival time for this cohort
    mean = cohort.cohortOutcomes.meanSurvivalTime

    # construct a normal likelihood
    # with mean calculated from the simulated data and standard deviation from the clinical study.
    # evaluate this pdf (probability density function) at the mean reported in the clinical study.
    log_l = stats.norm.logpdf(
        x=sets.OBS_MEAN,
        loc=mean,
        scale=sets.OBS_STDEV)

    return log_l


def calibrate_random_sampling(n_samples):
    """Run random sampling calibration with the specified prior ranges and log-likelihood function."""
    calib = CalibrationRandomSampling(prior_ranges=sets.PRIOR_RANGE)
    calib.run(log_likelihood_func=log_likelihood_func, num_samples=n_samples)
    calib.save_samples(file_name="output/samples_random.csv")


def calibrate_mcmc(n_samples, std_factor):
    """Run MCMC sampling calibration with the specified prior ranges and log-likelihood function."""
    calib = CalibrationMCMCSampling(prior_ranges=sets.PRIOR_RANGE)
    calib.run(log_likelihood_func=log_likelihood_func, std_factor=std_factor, num_samples=n_samples)
    calib.save_samples(file_name="output/samples_mcmc.csv")


def plot_posterior(method, n_resamples, weighted=True):
    """ plot the posterior distribution of the mortality probability """

    if method == 'random':
        calib = CalibrationRandomSampling(prior_ranges=sets.PRIOR_RANGE)
        calib.read_samples(file_name="output/samples_random.csv")
        calib.plot_posterior(
            n_resample=n_resamples, weighted=weighted, figsize=(5, 5),
            file_name='figs/calibration/{}_postr.png'.format(method))

    elif method == 'mcmc':
        calib = CalibrationMCMCSampling(prior_ranges=sets.PRIOR_RANGE)
        calib.read_samples(file_name="output/samples_mcmc.csv")
        calib.plot_posterior(
            n_warmup = int(0.1*n_resamples), figsize=(5, 5), parameter_names=['Mortality Probability'],
            file_name='figs/calibration/{}_postr.png'.format(method))

    else:
        raise ValueError("Unknown calibration method: {}".format(method))


def plot_mcmc_trace():
    """ plot the trace of the MCMC sampling """
    calib = CalibrationMCMCSampling(prior_ranges=sets.PRIOR_RANGE)
    calib.read_samples(file_name="output/samples_mcmc.csv")
    calib.plot_trace(
        figsize=(5, 5), file_name='figs/calibration/mcmc_trace.png',
        moving_ave_window=int(0.1 * sets.N_SAMPLES))