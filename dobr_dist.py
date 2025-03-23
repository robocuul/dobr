""" This module contains classes to calculate arrays of the pmf, cdf,
and loss function for fitted discrete distributions.

Copyright (c) 2020-2025 dr. R.A.C.M. Broekmeulen and dr. K.H. van Donselaar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""
# pylint: disable-msg=C0103
# For readability, we do not follow snake name convention
import abc
import math
import numpy as np
# pylint: disable=E0611
# Methods in SciPy are not always recognized by pylint
from scipy.special import binom, xlogy, xlog1py, gammaln

# Constants
EPS = 1.e-7
PANJER_EPS = 1.e-16
MAX_K = 500

def lb_vtm(mean):
    """Return the lower bound for the VTM of the demand
    for discrete (binominal) distributions.
    """
    trunc_mean = min(mean, MAX_K)
    k = int(trunc_mean)
    return (2 * k + 1) - ((k * (k + 1)) / trunc_mean) - trunc_mean

def select_dist(mean, vtm, trycp=False):
    """Determine appropiate mixture of discrete distributions,
    using the method of Adan et al (1995),
    and return the corresponding instance of the DistArray class.
    """
    if lb_vtm(mean) - vtm > EPS:
        # No discrete fit possible
        return None
    # Determine distribution type
    # Selection parameter theta = c2-1/mu = (vtm-1)/mu
    theta = (vtm - 1.) / mean
    if theta < -1. / MAX_K:
        # Binomial distribution
        return MixedBinomial(mean, vtm)
    if theta > 0. and trycp:
        # Compound (or stuttering) Poisson distribution
        return CompoundPoisson(mean, vtm)
    if theta >= 1.:
        # Geometric distribution
        return MixedGeometric(mean, vtm)
    if theta > 1. / MAX_K:
        # Negative binomial distribution
        return MixedNegBinomial(mean, vtm)
    # Poisson distribution (Theta=0)
    return FittedPoisson(mean, vtm)

def two_moment_fit(mean, stdev):
    """Determine appropiate mixture of discrete distributions,
    using the method of Adan et al (1995),
    and return the corresponding instance of the DistArray class.
    """
    if mean < EPS:
        print("Zero or negative mean period demand not allowed!")
        return None
    vtm = (stdev**2)/mean
    if lb_vtm(mean) - vtm > EPS:
        print("StDev of demand too low for corresponding mean")
        return None
    return select_dist(mean, vtm)

def _pmf_poisson(x, lrate, prev_pmf=-1.):
    """Calculate for the poisson distribution the pmf value for x,
    by using the Panjer recursion if possible.
    """
    if prev_pmf < PANJER_EPS:
        if x >= 0:
            logpmf = xlogy(x, lrate)-gammaln(x+1)-lrate
            return math.exp(logpmf)
        return 0.
    # Apply Panjer recursion
    return (lrate/x)*prev_pmf

def _pmf_negbinomial(x, t, p, prev_pmf=-1.):
    """Calculate for the negative binomial distribution the pmf value for x,
    by using the Panjer recursion if possible.
    """
    if prev_pmf < PANJER_EPS:
        if x >= 0:
            # Note: BinomLN(N,K) = GammaLN(N+1)-GammaLN(K+1)-GammaLN(N-K+1)
            # with Binom(T+X-1,X) we get
            binom_coeff = gammaln(t+x)-gammaln(x+1)-gammaln(t)
            log_pmf = binom_coeff + x*math.log(p) + xlog1py(t, -p)
            return math.exp(log_pmf)
        return 0.
    # Apply Panjer recursion
    return p*(1+(t-1)/x)*prev_pmf

def check_pmf(pmf):
    """ Check and repair the pmf."""
    if any(x < 0. for x in pmf):
        # We need to correct the pmf
        correction = 0.
        x_max = -1
        y_max = 0.
        for x in range(pmf.size):
            if pmf[x] < 0.:
                # print("Neg pmf", x, res_ioh[x], correction)
                correction += pmf[x]
                pmf[x] = 0.
            elif pmf[x] > y_max:
                y_max = pmf[x]
                x_max = x
        # print("Correction", correction, x_max, y_max)
        pmf[x_max] -= correction
    # if any(x < 0. for x in res_ioh):
    #     print("Repair failed")
    delta = 1.0-np.sum(pmf)
    if abs(delta) > EPS:
        if x_max > -1:
            pmf[x_max] += delta
        else:
            pmf[0] += delta
    return pmf


class DistArray(abc.ABC):
    """ An abstract class used to describe discrete distributions."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, mean, vtm, name="?"):
        self.mean = mean
        self.vtm = vtm
        self.name = name
        # Initialy, no error and no arrays
        self.dist_error = 0
        self.apmf = None
        self.acdf = None
        self.appf = None
        self.aloss = None

    def _create_dep_arrays(self):
        """Create the cdf, ppf, and loss arrays based on the pmf."""
        # Cdf array
        self.acdf = np.cumsum(self.apmf)

        # Ppf array
        self.appf = None

        # Loss array
        x = np.arange(self.apmf.size)
        acumpartial = np.cumsum(np.multiply(x, self.apmf))
        self.aloss = self.mean - acumpartial - x*(1.0-self.acdf)

    def create_ppf_array(self):
        """Create the ppf array based on the cdf."""
        self.appf = np.zeros(self.apmf.size)
        b_max = self.ub()
        x_lb = 0
        b = 0
        while b < b_max:
            while self.acdf[x_lb+1] < b/b_max:
                x_lb += 1
            self.appf[b] = x_lb
            b += 1
        self.appf[b_max] = b_max-1

    def dist_name(self):
        """Return the name of the fitted distribution."""
        return self.name

    def ub(self):
        """Return the upper bound of the pmf."""
        return self.apmf.size-1

    def pmf(self, x):
        """Return the pmf."""
        if 0 <= x <= self.ub():
            return self.apmf[x]
        # Past the range of the pmf: pmf=0
        return 0.

    def cdf(self, x):
        """Return the cdf."""
        if x < 0:
            return 0.
        if x <= self.ub():
            return self.acdf[x]
        # Past the range of the pmf: cdf=1
        return 1.

    def sf(self, x):
        """Return the survival function P[X>=x]."""
        if x <= 0:
            return 1.
        if x-1 <= self.ub():
            return 1. - self.acdf[x-1]
        # Past the range of the pmf: sf=0
        return 0.

    def loss(self, x):
        """Return the discrete loss E[(X-x)+]."""
        if x < 0:
            return self.mean
        if x <= self.ub():
            return self.aloss[x]
        # Past the range of the pmf: no loss
        return 0.

    def rloss(self, x):
        """Return the reversed discrete loss E[(x-X)+]."""
        # Since (a)+ = a + (-a)+, we have rloss(x) = x - E[X] + loss(x)
        if x < 0:
            return x
        if x <= self.ub():
            return self.aloss[x] + x - self.mean
        # Past the range of the pmf: no loss
        return x - self.mean

    def getvariate(self, univar):
        """Return the variate based on the cdf."""
        # Use bisection on cdf for inverse-transform method
        lowx = 0
        lowy = self.cdf(lowx)
        highx = self.ub()
        #highy = 1.

        while highx - lowx > 1:
            testx = lowx + int((highx-lowx)/2)
            testy = self.cdf(testx)
            if testy < univar:
                lowx = testx
                lowy = testy
            else:
                highx = testx
                #highy = testy

        if univar < lowy:
            return lowx
        return highx

    def getvariateppf(self, univar):
        """Return the variate based on the cdf."""
        # Get the lower and upper bound for the bisection, based on the ppf
        x_range = self.ub()
        x_lb = int(univar*x_range)
        lowx = int(self.appf[x_lb])
        lowy = self.cdf(lowx)
        highx = int(self.appf[min(x_range, x_lb + 1)]+1)

        # Use bisection on cdf for inverse-transform method
        while highx - lowx > 1:
            testx = lowx + int((highx-lowx)/2)
            testy = self.cdf(testx)
            if testy < univar:
                lowx = testx
                lowy = testy
            else:
                highx = testx
                #highy = testy

        if univar < lowy:
            return lowx
        return highx

class ZeroDistArray(DistArray):
    """ A class used to describe generated discrete distributions."""

    def __init__(self):
        super().__init__(0., 0., name="Zero")
        self.apmf = np.asarray([1.])
        # Create cdf and loss arrays
        self._create_dep_arrays()


class GeneratedDistArray(DistArray):
    """ A class used to describe generated discrete distributions."""

    def __init__(self, mean, vtm, apmf):
        super().__init__(mean, vtm, name="Generated")
        self.apmf = apmf
        # Create cdf and loss arrays
        self._create_dep_arrays()


class EmpiricalDistArray(DistArray):
    """ A class used to describe empirical discrete distributions."""

    def __init__(self, epmf):
        # Init with mean and vtm zero
        super().__init__(0., 0., name="Empirical")
        # Check for negative values in list
        if any(x < 0. for x in epmf):
            self.dist_error = -9901
            print("EPMF error: negative pmf values")
            # Reset mean and vtm to zero
            self.mean = 0.
            self.vtm = 0.
        if self.dist_error == 0:
            # Convert the list to a numpy array
            self.apmf = np.asarray(epmf)
            # Sum the pmf array
            sum_pmf = np.sum(self.apmf)
            # Check if the sum of the probabilities in the pmf
            # is equal to 1.
            if abs(sum_pmf - 1.) > EPS:
                self.dist_error = -9902
                print("EPMF error: Sum <> 1")
                # Reset mean and vtm to zero
                self.mean = 0.
                self.vtm = 0.
            else:
                # Determine mean and variance from the pmf
                # Create X array
                x = np.arange(self.apmf.size)
                # Calculate the mean = E[X]
                self.mean = np.dot(x, self.apmf)
                if self.mean == 0.:
                    # Reset vtm to zero
                    self.vtm = 0.
                else:
                    # Calculate the variance/mean
                    # (E[X**2]-E[X]**2)/E[X] = E[X**2]/E[X] - E[X]
                    self.vtm = np.dot(np.float_power(x, 2),
                        self.apmf)/self.mean-self.mean
                    # Create cdf and loss arrays
                    self._create_dep_arrays()


class FittedDistArray(DistArray):
    """ A class used to describe fitted discrete distributions."""

    def __init__(self, mean, vtm, name="Fitted"):
        super().__init__(mean, vtm, name=name)
        self.theta = (self.vtm - 1.) / self.mean
        self.q = 1.
        # Determine parameters
        self._set_param()
        # Create the arrays
        self.apmf = self._create_pmf_array()
        self._create_dep_arrays()

    @abc.abstractmethod
    def _set_param(self):
        pass

    @abc.abstractmethod
    def _pmf(self, x, first=True, prev_pmf=-1.):
        pass

    def _create_pmf_array(self):
        """Create the pmf array for mixed discrete distributions."""
        # Start with an empty list, zero cdf and x value zero
        tmp = []
        cdf = 0.
        x = 0
        # No previous values for the mixed pmf (required for Panjer recursions)
        pmf1 = -1.
        pmf2 = -1.
        # Add pmf values to the list
        while cdf < 1.0-EPS:
            pmf1 = self._pmf(x, first=True, prev_pmf=pmf1)
            if self.q < 1.:
                pmf2 = self._pmf(x, first=False, prev_pmf=pmf2)
                pmf = self.q*pmf1+(1.-self.q)*pmf2
            else:
                pmf2 = -1.
                pmf = pmf1
            # Append value to the list and update the cdf
            tmp.append(pmf)
            cdf += pmf
            x += 1
        # Padding needed (to make cdf=1 at the upper bound)?
        if cdf < 1.0:
            pmf = 1.0 - cdf
            tmp.append(pmf)
        # Return the list as numpy array
        return np.array(tmp)


class MixedBinomial(FittedDistArray):
    """ A class used to describe fitted binomial distributions."""

    def __init__(self, mean, vtm):
        super().__init__(mean, vtm, name="Binomial")

    def _set_param(self):
        # Determine parameters
        if self.mean / (1. - self.vtm) == int(self.mean / (1. - self.vtm)):
            # Pure binomial
            self.k = self.mean / (1. - self.vtm)
            self.p1 = 1. - self.vtm
            self.q = 1.
        else:
            self.k = int(-1. / self.theta)
            if self.k == 0:
                self.k = 1
            if abs(1. + self.theta) < EPS:
                self.q = 1.
            else:
                self.q = ((1. + self.theta * (1 + self.k)
                           +math.sqrt(-self.theta*self.k*(self.k + 1)
                                      -self.k)) / (1. + self.theta))
            self.p1 = self.mean / (self.k+1-self.q)
        self.p2 = self.p1
        self.lrate = self.mean

    def _pmf(self, x, first=True, prev_pmf=-1.):
        # Calculate for the binomial distribution the pmf value for x,
        # by using the Panjer recursion if possible.
        if first:
            k = self.k
            p = self.p1
        else:
            k = self.k+1
            p = self.p2
        if prev_pmf < PANJER_EPS or p == 1.:
            if 0 <= x <= k:
                return binom(k, x)*(p**x)*((1.-p)**(k-x))
            return 0.
        # Apply Panjer recursion
        return (k-(x-1))*p/(x*(1-p))*prev_pmf


class FittedPoisson(FittedDistArray):
    """ A class used to describe fitted poisson distributions."""

    def __init__(self, mean, vtm):
        super().__init__(mean, vtm, name="Poisson")

    def _set_param(self):
        # Determine parameters
        self.p1 = 1.
        self.lrate = self.mean
        self.q = 1.

    def _pmf(self, x, first=True, prev_pmf=-1.):
        return _pmf_poisson(x, self.lrate, prev_pmf=prev_pmf)


class CompoundPoisson(FittedDistArray):
    """ A class used to describe fitted compound poisson distributions."""

    def __init__(self, mean, vtm):
        super().__init__(mean, vtm, name="Compound Poisson")

    def _set_param(self):
        # Determine parameters
        # using a shifted geometric distribution as compounding distribution
        self.p1 = 2. / (self.vtm + 1.)
        self.lrate = self.mean * self.p1
        self.q = 1.

    def _create_pmf_array(self):
        """Create the pmf array for compound Poission distributions."""
        # Create the compound Poisson distribution
        # Start with an empty list, zero cdf and x value zero
        tmp = []
        cdf = 0.
        x = 0
        pmf_poisson = []            # List with Poisson pmf values
        snb_prev = [-1.]            # List with last Shifted NegBin pmf values
                                    #   to facilitate Panjer recursion
        # Add pmf values to the list
        while cdf < 1.0-EPS:
            if x == 0:
                pmf_poisson.append(_pmf_poisson(x, self.lrate))
                pmf = pmf_poisson[0]
                last_poisson = 0
            else:
                pmf = 0.
                w = 1
                while w <= x:
                    if w > last_poisson:
                        pmf_poisson.append(_pmf_poisson(w, self.lrate,
                            prev_pmf=pmf_poisson[last_poisson]))
                        last_poisson += 1
                    if w > len(snb_prev)-1:
                        snb_prev.append(-1.0)
                    # New order size
                    snb_prev[w] = _pmf_negbinomial(x-w, w, 1.0-self.p1,
                                               prev_pmf=snb_prev[w])
                    pmf += pmf_poisson[w]*snb_prev[w]
                    w += 1
            # Append value to the list and update the cdf
            tmp.append(pmf)
            cdf += pmf
            x += 1
        # Padding needed (to make cdf=1 at the upper bound)?
        if cdf < 1.0:
            pmf = 1.0 - cdf
            tmp.append(pmf)
        # Return the list as numpy array
        return np.array(tmp)

    def _pmf(self, x, first=True, prev_pmf=-1.):
        # Dummy
        pass


class MixedNegBinomial(FittedDistArray):
    """ A class used to describe fitted negative binomial distributions."""

    def __init__(self, mean, vtm):
        super().__init__(mean, vtm, name="Negative Binomial")

    def _set_param(self):
        # Determine parameters
        self.k = int(1. / self.theta)
        self.q = (((1. + self.k) * self.theta
                   - math.sqrt((1. + self.k) * (1. - self.theta * self.k)))
                  / (1. + self.theta))
        self.p1 = self.mean / (self.k + 1. - self.q + self.mean)
        self.p2 = self.p1

    def _pmf(self, x, first=True, prev_pmf=-1.):
        # Calculate for the negative binomial distribution the pmf value for x,
        # by using the Panjer recursion if possible.
        if first:
            t = self.k
            p = self.p1
        else:
            t = self.k+1
            p = self.p2
        if prev_pmf < PANJER_EPS:
            if x >= 0:
                # Note: BinomLN(N,K) = GammaLN(N+1)-GammaLN(K+1)-GammaLN(N-K+1)
                # with Binom(T+X-1,X) we get
                binom_coeff = gammaln(t+x)-gammaln(x+1)-gammaln(t)
                log_pmf = binom_coeff + x*math.log(p) + xlog1py(t, -p)
                return math.exp(log_pmf)
            return 0.
        # Apply Panjer recursion
        return p*(1+(t-1)/x)*prev_pmf


class MixedGeometric(FittedDistArray):
    """ A class used to describe fitted geometric distributions."""

    def __init__(self, mean, vtm):
        super().__init__(mean, vtm, name="Geometric")

    def _set_param(self):
        # Determine parameters
        root_plus = 1. + self.theta + math.sqrt(self.theta * self.theta - 1.)
        root_min = 1. + self.theta - math.sqrt(self.theta * self.theta - 1.)
        self.q = 1. / root_plus
        self.p1 = 1.0 - self.mean * root_plus / (2. + self.mean * root_plus)
        self.p2 = 1.0 - self.mean * root_min  / (2. + self.mean * root_min)

    def _pmf(self, x, first=True, prev_pmf=-1.):
        # Calculate for the geometric distribution the pmf value for x,
        # by using the Panjer recursion if possible.
        if first:
            p = self.p1
        else:
            p = self.p2
        if prev_pmf < PANJER_EPS:
            if x >= 0:
                return p*((1.0-p)**x)
            return 0.
        # Apply Panjer recursion
        return (1-p)*prev_pmf
