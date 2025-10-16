import MultiSurvivalModelClasses as SurvivalCls
import calib_sets as sets
from deampy.calibration import CalibrationRandomSampling, CalibrationMCMCSampling
import numpy as np


class CalibratedModel:
    """ to run the calibrated survival model """

    def __init__(self, calib_method, samples_csv_filename=None,  drug_effectiveness_ratio=1):
        """ extracts seeds, mortality probabilities and the associated likelihood from
        the csv file where the calibration results are stored
        :param calib_method: (string) 'random' or 'mcmc'
        :param samples_csv_filename: (string) name of the csv file containing parameter samples
        :param drug_effectiveness_ratio: effectiveness of the drug
        """

        self.drugEffRatio = drug_effectiveness_ratio

        if calib_method == 'random':
            self.calib = CalibrationRandomSampling(prior_ranges=sets.PRIOR_RANGE)
        elif calib_method == 'mcmc':
            self.calib = CalibrationMCMCSampling(prior_ranges=sets.PRIOR_RANGE)
        else:
            raise ValueError("Unknown calibration method: {}".format(calib_method))

        file_name = "output/samples_{}.csv".format(calib_method) if samples_csv_filename is None else samples_csv_filename
        self.calib.read_samples(file_name=file_name)

        self.multiCohorts = None  # multi-cohort

    def simulate(self, num_of_simulated_cohorts, cohort_size, time_steps):
        """ simulate the specified number of cohorts based on their associated likelihood weight
        :param num_of_simulated_cohorts: number of cohorts to simulate
        :param cohort_size: the population size of cohorts
        :param time_steps: simulation length
        """

        if isinstance(self.calib, CalibrationRandomSampling):
            # resample the seeds and mortality probabilities
            self.calib.resample(n_resample=num_of_simulated_cohorts)

            seeds = self.calib.resampledSeeds

            # simulate the desired number of cohorts
            self.multiCohorts = SurvivalCls.MultiCohort(
                ids=range(num_of_simulated_cohorts),
                pop_sizes=[cohort_size] * num_of_simulated_cohorts,
                mortality_probs= np.array(self.calib.resamples['Mortality Probability']) * self.drugEffRatio)

        elif isinstance(self.calib, CalibrationMCMCSampling):

            seeds = self.calib.seeds[-num_of_simulated_cohorts:]

            # simulate the desired number of cohorts
            self.multiCohorts = SurvivalCls.MultiCohort(
                ids=range(num_of_simulated_cohorts),
                pop_sizes=[cohort_size] * num_of_simulated_cohorts,
                mortality_probs=np.array(self.calib.samples['Mortality Probability'][-num_of_simulated_cohorts:])
                                * self.drugEffRatio)
        else:
            raise ValueError("Unknown calibration method: {}".format(self.calib))

        # simulate all cohorts
        self.multiCohorts.simulate(time_steps, seeds=seeds)

    def get_mean_survival_time_proj_interval(self, alpha):
        """
        :param alpha: the significance level
        :returns tuple in the form of (mean, [lower, upper]) of projection interval
        """

        mean = self.multiCohorts.multiCohortOutcomes.statMeanSurvivalTime.get_mean()
        proj_interval = self.multiCohorts.multiCohortOutcomes.statMeanSurvivalTime.get_PI(alpha=alpha)

        return mean, proj_interval

