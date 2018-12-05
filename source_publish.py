import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class Population(object):
    def __init__(self, size, optimum, ratio, coefficient=1):
        """
        Create a population of specified size.
        
        Arguments:
        size -- size of the population (positive integer > 1)
        optimum -- optimal fitness (float)
        ratio -- relative amount of the most fit members selected for
            reproduction in each generation (positive float < 1)
            NOTE: If the resulting number of the selected most fit, i.e.
                int(ratio*self.size)+1, is not even we set it to be the closest
                larger even integer
        coefficient -- specifies the amount of randomness in the generation of
            the offspring (positive float)
        """

        self.size = size
        self.optimum = optimum
        self.select = max(int(ratio*self.size),2)
        if self.select%2 != 0:
            self.select += 1
        self.coefficient = coefficient
        self.members = []
        self.generations = 0
        
    def fillRandom(self, mu, sigma):
        """"Set fitness randomly using Gaussian distribution of mean mu and
        standard deviation sigma."""
        
        self.members = np.random.normal(mu, sigma,size)
    
    def unify(self, value):
        """Set fitness of all members to value"""
        self.members = value*np.ones(self.size)

    def changeMember(self, i, value):
        """Change the phenotype of i-th member."""
        try:
            self.members[i] = value
        except IndexError:
            print("Index out of range.")

    def getMean(self):
        """Return the mean of self.members"""
        return np.mean(self.members)

    def getVariance(self):
        """Return the variance of self.members"""
        return np.var(self.members)

    def getSTD(self):
        """Return the standard deviation of self.members"""
        return np.std(self.members)

    def getCoefVariation(self):
        """Return the coefficient of variation of self.members"""
        return stats.variation(self.members)

    def getMax(self):
        """Return the maximum fitness of the population"""
        return np.amax(self.members)

    def nextGeneration(self, generations=1):
        """
        Proceed to next generations (number specified by the parameter
        generations, implicitly 1) using the following rules:
            1.  The most fit self.select members will be randomly paired into
                self.select/2 pairs.
            2.  If self.select is divisible by the number of pairs, each pair
                produces 2*self.size/self.select descendants to keep the size of
                the population constant, otherwise each pair produces
                (2*self.size/self.select + 1) descendants. Each pair produces
                one descandant at a time. From the last round of generation a
                suitable number of descendatds is selected to keep the size of
                the population constant. The selection in the final round is
                implicitly random by selecting the first few of the descendants
                of randomly paired and ordered parents.
            3.  The fitness of the new descendant is determined by a Gaussian
                distribution with mean given by the mean of parental fitness and
                standard deviation defined by
                    self.coefficient*0.5*(distance of parental fitness).
        """
        
        for generation in xrange(generations):
            self.generations += 1
            # Compute handicaps
            handicaps = np.absolute((self.optimum - self.members)**2)
            # Sort self.members by handicaps by finding and applying the proper
            # sorting permutation p
            p = handicaps.argsort()
            sortedMembers = self.members[p]
            # Select the most fit self.select members and order them randomly
            selected = sortedMembers[0:self.select]
            np.random.shuffle(selected)
            # Create a shifted copy of the selected memebers to simplify the
            # prescription for new generation
            shifted = np.roll(selected, -1)
            # Recall that self.select might not be divisible by the number of
            # pairs. To compensate generate more descendants to keep the size of
            # the population constant.
            for i in range((2*self.size/self.select)+1):
                # Produce self.select new members
                offspring = self.coefficient*0.5*np.absolute(selected - shifted)\
                            *np.random.randn(self.select)\
                            + 0.5*(selected + shifted)
                # Pick the ones parented by the right pair of parents
                offspring = offspring[::2]
                # Add offspring to members; the "if" clause required as
                # np.concanate does not behave well with an empty array
                if i == 0:
                    self.members = offspring
                else:
                    self.members = np.concatenate((self.members, offspring))
            # Leave out residual offspring
            self.members = self.members[:self.size:]        

def example():
    """Example method that sets up a population of one sport among uniform
    population, runs for 100 generations and plots the relevant statistical
    data."""
    
    # For simplicity we first list all the experiment data
    popsize = 324000
    survival_ratio = 0.2
    coef_nu = 0.65
    generations = 500
    optimum = 120
    initial_fitness = 10
    sport_fitness = 11
    
    # Set up a new population of popsize with optimum, ratio of survival and the
    # coefficient nu defined above
    population = Population(popsize, optimum, survival_ratio, coef_nu)
    # Prescribe the uniform fitness
    population.unify(initial_fitness)
    # Insert a sport (its position does not play any role)
    population.changeMember(0, sport_fitness)

    # Set up statistical data storage
    means = np.array([population.getMean()])
    variances = np.array([population.getVariance()])
    stds = np.array([population.getSTD()])
    coefsVariation = np.array([population.getCoefVariation()])
    # Number of the first generation in which the optimum is achieved
    # For the purpose of this example, the optimum is achieved once there exists
    # a member such that his fitness is larger than (the optimum - 1)
    firstOptimal = None

    # Run for 30 generations and store relevant statistical data
    for generation in xrange(generations):
        # Proceed to the next generation
        population.nextGeneration()
        # Check whether the optimum has been achieved in the above sense
        if firstOptimal == None and population.getMax() > (population.optimum - 1):
            firstOptimal = generation
        # Store relevant data
        means = np.append(means, np.asarray([population.getMean()]))
        variances = np.append(variances, np.asarray([population.getVariance()]))
        stds = np.append(stds, np.asarray([population.getSTD()]))
        coefsVariation = np.append(coefsVariation,\
                                   np.asarray([population.getCoefVariation()]))

    # Plot the results using pyplot
    plt.title('Exemplar with ' + 'N = '+str(popsize) + ', s = '+str(survival_ratio) + ', ' + r'$\nu$ = '+str(coef_nu), loc='left', fontsize=15)
    plt.xlabel("Generation")
    plt.ylabel("Trait value relative to the optimum")
    plt.ylim(0, 1.2)
    plt.plot(range(generations+1), means/population.optimum, color="r", label="Relative mean")
    plt.plot(range(generations+1), (means+2*stds)/population.optimum, color="#eb8871", label='+/-2 standard deviations')
    plt.plot(range(generations+1), (means-2*stds)/population.optimum, color="#eb8871")
    plt.plot(range(generations+1), coefsVariation, color="b", label="Coefficient of Variation")
    plt.fill_between(range(generations+1), (means-2*stds)/population.optimum,\
                     (means+2*stds)/population.optimum, hatch = '///', color="#fbe0d5")
    if firstOptimal != None:
        plt.plot((firstOptimal, firstOptimal), (0, 1.2), 'green', label="Optimum achieved")
    plt.legend(loc="best")
    plt.show()
