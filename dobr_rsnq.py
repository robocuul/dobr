# pylint: disable-msg=C0302
""" This module contains classes to calculate key performance indicators for
periodic inventory control systems.

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
# For readability, we do not always follow snake name convention
# pylint: disable-msg=R0902
# pylint: disable-msg=R0904
# pylint: disable-msg=R0911
# pylint: disable-msg=R0912
# pylint: disable-msg=R0913
# pylint: disable-msg=R0914
# pylint: disable-msg=R0915
# pylint: disable-msg=R0916

import abc
import math
import json
import numpy as np
from scipy.stats import gamma, norm
from scipy.signal import convolve
from scipy.optimize import brentq
import dobr_dist

# Constants
VERSION = 250214
EPS = 1.e-7             # Default accuracy (epsilon)
DEF_TARGET = 0.95       # Default service target for the reorder level
FCC_UMAX = 3

# Constants for the accuracy of the results
VAL_EXACT = 0           # Result is exact
VAL_APPROX = 1          # Results is based on an approximation
VAL_CI = 2              # Results is based on an approximation and in the 
                        # confidence interval (CI) of the simulation
VAL_NOTAPPLIC = 3       # KPI is not applicable for the given parameters
VAL_NOTYET = 4          # No results are available yet
VAL_ERROR = 5           # No result, due to input errors
VAL_EMPTY = 6           # No result yet calculated
# List of error codes
ERROR_CODES = {
    -9900 : "Internal inconsistency",
    -9901 : "Negative values in PMF",
    -9902 : "Sum PMF <> 1!",
    -9903 : "StDev of demand too low for corresponding mean",
    -9904 : "Negative lead time not allowed!",
    -9905 : "Negative stdev lead time not allowed!",
    -9906 : "Zero or negative review period not allowed!",
    -9907 : "Zero or negative mean period demand not allowed!",
    -9908 : "Zero variance period demand not allowed!",
    -9909 : "Lead time must be less or equal to reviewperiod",
    -9911 : "Negative IOQ not allowed!",
    -9912 : "Non-integer IOQ not allowed!",
    -9913 : "Zero or negative Minimal Order Quantity (MOQ) not allowed!",
    -9914 : "Non-integer Minimal Order Quantity (MOQ) not allowed!",
    -9915 : "MOQ must be a integer multiple of the IOQ!",
    -9925 : "Unit loads less than zero are not allowed!",
    -9926 : "Unit loads must be a integer multiple of the IOQ!",
    -9927 : "Negative capacity not allowed!",
    -9928 : "Zero or negative target not allowed!",
    -9929 : "Target must be less than 1.0!",
    -9980 : "No zreg_fifo file found",
    -9981 : "Singular matrix for IP distribution",
    -9982 : "Singular matrix for IOH distribution",
    -9930 : "Shelflife must be greater than the review period!",
    -9931 : "Non-integer shelflife not allowed!",
    -9999 : "Not yet implemented/available"}
# The default input data in case no input file is found
DEFAULT_DATA = {
    "distribution" : "Discrete",
    "lostsales" : False,
    "leadtime" : 1,
    "stdev_leadtime" : 0,
    "reviewperiod" : 1,
    "mean_perioddemand" : 1,
    "stdev_perioddemand" : 1,
    "ioq" : 1,
    "moq" : 0,
    "reorderlevel" : 1,
    "shelfspace" : 0,
    "unitcap" : 0,
    "shelflife" : 0,
    "EWA" : True,
    "ROS" : False,
    "fifo" : 1.,
    "concurrent" : True}
KPI_CATALOGUE = {
    "Fillrate" : "Fill rate P2",
    "Readyrate" : "Discrete ready rate P3D",
    "ELT" : "Effective lead-time",
    "EBO_L" : "Expected backorders begin E[BO(L)]",
    "EBO_RL" : "Expected backorders end E[BO(R+L)]",
    "EIOH_L" : "Expected inventory begin E[IOH(L)]",
    "EIOH_RL" : "Expected inventory end E[IOH(R+L)]",
    "EOL" : "Expected order lines E[OL]",
    "EOS" : "Expected order size E[OS]",
    "ESUP" : "Expected supply",
    "EST" : "Expected sojourn time E[ST]",
    "EW" : "Expected outdating E[W]",
    "EUA" : "Expected unit load arrivals E[UA]",
    "ENB" : "Expected # batches on stock E[NB]",
    "EUSL_L" : "Expected # of locations E[USL(L)]",
    "POC" : "Probability shelf overflow P[X>V]",
    "EIBR" : "Expected backroom inventory E[IBR]",
    "ENIR" : "Expected idle backroom trips E[NIR]",
    "ENCR" : "Expected conc. backroom trips E[NCR]"}

def validate_input(value,
                   noneg_val=False, nozero_val=False, pos_val=False,
                   int_val=False, ub_1=False, ub_1eps=False):
    """ Generic validator for DoBr functions."""
    # Is the input a number?
    try:
        # Convert the content to a float
        input_value = float(value)
        # Check the requested conditions
        if ((noneg_val and input_value < 0.)
            or (nozero_val and abs(input_value) < EPS)
            or (pos_val and input_value < EPS)
            or (int_val and int(input_value) != input_value)
            or (ub_1eps and input_value > (1.0-EPS))
            or (ub_1 and input_value > 1.0)):
            return False
        return True
    except ValueError:
        # Error = NaN
        return False

def applicable_kpi(kpi, data_dict):
    """ Determine if the KPI is applicable based on the input data."""
    if kpi in ("EIOH_Cont", "NOO", "Sales", "Demand"):
        # No counterpart and/or in dashboard
        return False
    if (data_dict['lostsales']
        and (kpi in ("EBO_L", "EBO_RL", "EBO_Cont"))):
        return False
    if data_dict['stdev_leadtime'] == 0 and kpi == "ELT":
        return False
    if data_dict['shelflife'] == 0 and kpi == "EW":
        return False
    if (data_dict["shelfspace"] == 0
          and (kpi in ("POC", "EIBR", "ENIR", "ENCR"))):
        return False
    return True

def not_error_code(value):
    """ Check if returned KPI value is not an error code."""
    if value > -9900:
        return True
    return False

def print_data_header(data_dict):
    """ Summarize the input data in a print header."""
    if data_dict["lostsales"]:
        print(" KPI's for a periodic review inventory system"
              + " assuming lost sales")
    else:
        print(" KPI's for a periodic review inventory system"
              + " assuming backordering")
    print(f"  and {data_dict['distribution']} distributed demand"
          + f" (Mean={round(data_dict['mean_perioddemand'], 3):g}, "
          + f"StDev={round(data_dict['stdev_perioddemand'],3):g}),")
    line = f"  L={round(data_dict['leadtime'],3):g}"
    if data_dict['stdev_leadtime'] > EPS:
        line = line + f", StDevL={round(data_dict['stdev_leadtime'],3):g},"
    else:
        line = line + " (fixed),"
    line = line + f" R={round(data_dict['reviewperiod'],3):g},"
    line = line + f" IOQ={data_dict['ioq']},"
    if data_dict['moq'] > data_dict['ioq']:
        line = line + f" MOQ={data_dict['moq']},"
    if len(line) > 65:
        print(line)
        line = " "
    if data_dict['unitcap'] > EPS:
        line = line + f" U={data_dict['unitcap']},"
    if len(line) > 65:
        print(line)
        line = " "
    if data_dict['shelfspace'] > EPS:
        line = line + f" V={data_dict['shelfspace']},"
    if len(line) > 65:
        print(line)
        line = " "
    if data_dict['shelflife'] > 0:
        line = line + f" m={data_dict['shelflife']},"
    if len(line) > 65:
        print(line)
        line = " "
    line = line + f" and s={data_dict['reorderlevel']}"
    print(line)
    print(" ".ljust(70, "-")+" ")

def print_kpis_ana(kpi_results, data_dict, sku_name=None):
    """Print the analytic obtained KPI's of the inventory system."""

    if sku_name is None:
        print("*= DoBr analytic output ".ljust(70, "=")+"*")
    else:
        print(("*= DoBr analytic output " + sku_name + " ").ljust(70, "=")+"*")
    print_data_header(data_dict)
    for kpi in kpi_results:
        value = kpi_results[kpi][0]
        if not_error_code(value):
            if (kpi_results[kpi][1] != VAL_NOTYET
                and kpi_results[kpi][1] != VAL_NOTAPPLIC):
                if kpi_results[kpi][1] == VAL_EXACT:
                    acc_str = ""
                else:
                    acc_str = "~"
                print(f" {KPI_CATALOGUE[kpi].ljust(36)}"
                  + f" : {value: 12.3f}"+acc_str)
    print("*".ljust(70, "=")+"*")

def stdev_powerlaw(mean):
    """Return the standard deviation of the demand using the power law."""
    return max(mean**0.5, 1.18 * (mean ** 0.77))

def validate_discrete_vtm(mean, stdev, leadtime, reviewperiod):
    """Return the error message for the validation of the lower bound
    on the variance-to-mean ratio.
    """
    vtm = (stdev**2)/mean
    if vtm < 1.0:
        if EPS < leadtime < reviewperiod:
            min_p = leadtime
            fit_text = "L"
        else:
            min_p = reviewperiod
            fit_text = "R"
        lb_min_p = dobr_dist.lb_vtm(min_p*mean)
        if lb_min_p - vtm > EPS:
            # No fit
            #self.error_param[sku] = True
            min_stdev = (lb_min_p*mean)**0.5
            return False, (": StDev too low for mean of period " + fit_text
                      + f" (LB= {min_stdev})")
        return True, ""
    return True, ""

def absorbing_states(p):
    """ Detect possible absorbing states. """
    # Count the number of absorbing states in the transition matrix p
    states = 0
    for i in range(len(p)):
        if p[i, i] == 1. and np.sum(p[i, :]) == 1.:
            states += 1
    return states


class InvSys(abc.ABC):
    """ An abstract class used to describe periodic review inventory systems.

    Note that this parent class has some abstract methods, since the demand
    distribution is only specified in the child or subclasses!
    """

    def __init__(self, mean_perioddemand, stdev_perioddemand, leadtime,
                 stdev_leadtime=0., reviewperiod=1.0, ioq=1, moq=0, shelflife=0,
                 sub_error=0):
        """ Initialize class variables:
        mean_perioddemand   -- mean demand per period (> 0)
        stdev_perioddemand  -- standard deviation of the demand per period (> 0)
        leadtime            -- time between ordering and delivery in periods

        Keyword arguments:
        stdev_leadtime      -- the standard deviation of the leadtime (default 0.0)
        reviewperiod        -- the time between two reviews (default 1.0)
        ioq                 -- the incremental order quantity (default 1)
        moq                 -- the minimal order quantity (default 0)
        shelflife           -- the remaining shelflife upon entering the stock point
        sub_error           -- possible error codes from sub classes
        """
        # Still free from validation errors (from subclasses)?
        self.dobr_error = sub_error
        # Assign parameters to attributes and check validity
        if self.dobr_error == 0 and validate_input(leadtime, noneg_val=True):
            self.leadtime = leadtime
        else:
            self.dobr_error = -9904
        if self.dobr_error == 0 and validate_input(reviewperiod, pos_val=True):
            self.reviewperiod = reviewperiod
        else:
            self.dobr_error = -9906
        if self.dobr_error == 0 and validate_input(mean_perioddemand, pos_val=True):
            self.mean = mean_perioddemand
        else:
            self.dobr_error = -9907
        if self.dobr_error == 0 and validate_input(stdev_perioddemand,
                                                   nozero_val=True):
            if stdev_perioddemand < -EPS:
                # Apply power law
                self.variance = stdev_powerlaw(self.mean)**2
            else:
                self.variance = stdev_perioddemand**2
            # Determine the Variance-To-Mean (VTM)
            self.vtm = self.variance/self.mean
        else:
            self.dobr_error = -9908
        if self.dobr_error == 0 and validate_input(ioq, noneg_val=True):
            self.ioq = ioq
        else:
            self.dobr_error = -9911
        if self.dobr_error == 0 and validate_input(moq, noneg_val=True):
            self.moq = moq
        else:
            self.dobr_error = -9913
        if self.dobr_error == 0 and validate_input(shelflife, noneg_val=True):
            self.shelflife = shelflife
        else:
            self.dobr_error = -9930
        if self.dobr_error == 0 and validate_input(stdev_leadtime, noneg_val=True):
            self.stdev_leadtime = stdev_leadtime
            # Determine lead-time variance
            if self.stdev_leadtime > EPS:
                # Approximation of the standard deviation of the ELT
                # (=Effective Lead time) according to Bischak et al (2014),
                # with an extension to order quantities greater than 1
                tbo = max(self._max_oq(), self.reviewperiod*self.mean)/self.mean
                self.var_leadtime = (self.stdev_leadtime
                    *(1.0-0.8758*math.exp(-1.0898*tbo/self.stdev_leadtime)))**2
                # print("TBO", tbo, "Eff StDev LT", math.sqrt(self.var_leadtime))
            else:
                self.var_leadtime = 0.
        else:
            self.dobr_error = -9905

        # Default distribution is not discrete
        self.discrete = False
        self.lost_sales = False

    def _val_unitcap(self, unitcap):
        # Check the value of the unit capacity
        if self.dobr_error == 0:
            if validate_input(unitcap, noneg_val=True):
                ioq_mult = unitcap/max(1.0, self.ioq)
                if abs(ioq_mult - int(ioq_mult)) > EPS:
                    self.dobr_error = -9926
                    return False
                return True
            self.dobr_error = -9925
            return False
        return False

    def _val_shelfspace(self, capacity):
        # Check the value of the (shelf) capacity
        if self.dobr_error == 0:
            if validate_input(capacity, noneg_val=True):
                return True
            self.dobr_error = -9927
            return False
        return False

    def _val_target(self, target):
        if self.dobr_error == 0:
            if validate_input(target, pos_val=True):
                if validate_input(target, ub_1eps=True):
                    return True
                self.dobr_error = -9929
                return False
            self.dobr_error = -9928
            return False
        return False

    def _max_oq(self):
        return max(self.ioq, self.moq)

    @staticmethod
    def adjust_rol(reorderlevel):
        """ Default reorder level assumes continuous demand. """
        return float(reorderlevel)

    def mioh(self, reorderlevel):
        """ Return the maximum inventory on hand for continous demand. """
        return reorderlevel + self.ioq

    # @staticmethod
    def _adjust_shelfspace(self, capacity):
        """ Default capacity assumes continuous demand. """
        return float(capacity)

    def _baseperiods(self, cycle):
        if cycle == "R":
            return self.reviewperiod
        if cycle == "L":
            return self.leadtime
        if cycle == "RL":
            return self.reviewperiod+self.leadtime
        if cycle == "RLm1":
            return max(self.leadtime+1,self.reviewperiod-1+self.leadtime)
        if cycle == "Rm1":
            # Replacement for RLm1 in case of lost sales
            return max(1,self.reviewperiod-1)
        if cycle == "RLmid":
            return max(self.leadtime+1,int((self.reviewperiod-1)/2)+self.leadtime)
        if cycle == "Rmid":
            # Replacement for RLmid in case of lost sales
            return max(1,int((self.reviewperiod-1)/2))
        if cycle == "Rfrac":
            if self.leadtime == self.reviewperiod:
                return 0.
            l_over_r = self.leadtime/self.reviewperiod
            return (self.reviewperiod
                    -(self.leadtime-self.reviewperiod*int(l_over_r)))
        if cycle == "L1R":
            l_over_r = self.leadtime/self.reviewperiod
            return (int(l_over_r)+1)*self.reviewperiod
        if cycle == "Rho":
            return int(self.shelflife/self.reviewperiod)*self.reviewperiod
        if cycle == "LM":
            return self.leadtime+self.shelflife
        if cycle == "Mm0":
            return self.shelflife
        if cycle == "Mm1":
            return self.shelflife-1
        if cycle == "Mm2":
            return self.shelflife-2
        if cycle == "Mm3":
            return self.shelflife-3
        return -9900

    def _mu_cycle(self, cycle):
        return self.mean*self._baseperiods(cycle)

    def _vtmplus(self, cycle):
        # Calculate the additional variance-to-mean (VTM)
        # in case of stochastic lead-time
        if self.leadtime > EPS and self.var_leadtime > EPS:
            if cycle == "L":
                return self.mean*self.var_leadtime/self.leadtime
            if cycle == "RL":
                return (self.mean*self.var_leadtime
                        /(self.reviewperiod+self.leadtime))
            if cycle == "L1R":
                l_over_r = self.leadtime/self.reviewperiod
                return (self.mean*self.var_leadtime
                        /(int(l_over_r)+1)*self.reviewperiod)
            if cycle == "LM":
                return (self.mean*self.var_leadtime
                        /(self.shelflife+self.leadtime))
            return 0.
        return 0.

    @abc.abstractmethod
    def accuracy(self):
        """ Accuracy indication. """

    @abc.abstractmethod
    def _prob_pos_demand(self, capacity):
        pass

    @abc.abstractmethod
    def _ebo(self, cycle, reorderlevel, eioh=None, acc=False):
        pass

    @abc.abstractmethod
    def _eioh(self, cycle, reorderlevel, capacity=0, acc=False):
        pass

    @abc.abstractmethod
    def _poc(self, cycle, reorderlevel, capacity=0, acc=False):
        pass

    @abc.abstractmethod
    def _eol(self, reorderlevel, acc=False):
        pass

    @abc.abstractmethod
    def _eua(self, reorderlevel, unitcap, acc=False):
        pass

    @abc.abstractmethod
    def _enb(self, reorderlevel, acc=False):
        pass

    @abc.abstractmethod
    def _eusl(self, reorderlevel, unitcap, eos, acc=False):
        pass

    @abc.abstractmethod
    def _encr(self, reorderlevel, capacity, acc=False):
        pass

    @abc.abstractmethod
    def _ez(self, reorderlevel, acc=False):
        pass

    @abc.abstractmethod
    def _supply_factor(self, reorderlevel, acc=False):
        pass

    # The basic KPIs: each method checks if the class initialized without
    # errors (dobr_error == 0) and has correct additional parameters
    def fillrate(self, reorderlevel, acc=False):
        """Return the fill rate."""
        if self.dobr_error == 0:
            reorderlevel = self.adjust_rol(reorderlevel)
            fr = self._fr(reorderlevel)
            if acc:
                return fr, self.accuracy()
            return fr
        return self.dobr_error

    def _fr(self, reorderlevel, eioh_l=None, eioh_rl=None):
        # Fill rate = (E[IOH_L] - E[IOH_RL])/E[D_R]
        if eioh_l is None:
            eioh_l = self._eioh("L", reorderlevel)
        if eioh_rl is None:
            eioh_rl = self._eioh("RL", reorderlevel)
        return max(0., (eioh_l-eioh_rl)/(self.reviewperiod*self.mean))

    def readyrate(self, reorderlevel, acc=False):
        """Return the discrete ready rate."""
        if self.dobr_error == 0:
            reorderlevel = self.adjust_rol(reorderlevel)
            # Discrete ready rate = P[D_RL>s]
            return self._poc("RL", reorderlevel, acc=acc)
        return self.dobr_error

    def servdiffrate(self, reorderlevel, acc=False):
        """Return the service differentiation rate."""
        if self.dobr_error == 0:
            reorderlevel = self.adjust_rol(reorderlevel)
            sdr = max(0., self._poc("RL", reorderlevel)
                    /self._poc("L", reorderlevel))
            if acc:
                return sdr, self.accuracy()
            return sdr
        return self.dobr_error

    def ebo_l(self, reorderlevel, acc=False):
        """Return the expected backorders after the lead-time."""
        if self.dobr_error == 0:
            reorderlevel = self.adjust_rol(reorderlevel)
            return self._ebo("L", reorderlevel, acc=acc)
        return self.dobr_error

    def ebo_rl(self, reorderlevel, acc=False):
        """ Return the expected backorders after review period
        plus lead-time.
        """
        if self.dobr_error == 0:
            reorderlevel = self.adjust_rol(reorderlevel)
            return self._ebo("RL", reorderlevel, acc=acc)
        return self.dobr_error

    def eioh_l(self, reorderlevel, acc=False):
        """ Return the expected inventory-on-hand after the lead-time."""
        if self.dobr_error == 0:
            reorderlevel = self.adjust_rol(reorderlevel)
            return self._eioh("L", reorderlevel, acc=acc)
        return self.dobr_error

    def eioh_rl(self, reorderlevel, acc=False):
        """ Return the expected inventory-on-hand after review period
        plus leadtime.
        """
        if self.dobr_error == 0:
            reorderlevel = self.adjust_rol(reorderlevel)
            return self._eioh("RL", reorderlevel, acc=acc)
        return self.dobr_error

    def _eioh_c(self, reorderlevel, eioh_l):
        # Return average expected inventory-on-hand after review period
        # plus leadtime minus 1 period.
        if self.reviewperiod == 1:
            eioh_c = eioh_l
        else:
            eioh_rl1 = self._eioh("RLm1", reorderlevel)
            if self.reviewperiod < 3:
                # Trapezium rule
                eioh_c = (eioh_l+eioh_rl1)/2.
            else:
                # Simpson's rule
                eioh_m = self._eioh("RLmid", reorderlevel)
                eioh_c = (eioh_l+4*eioh_m+eioh_rl1)/6.
        return eioh_c

    def eol(self, reorderlevel=0, acc=False):
        """Return the expected number of order lines per review period.
        The reorder level is only required for the lost sales situation.
        """
        if self.dobr_error == 0:
            reorderlevel = self.adjust_rol(reorderlevel)
            return self._eol(reorderlevel, acc=acc)
        return self.dobr_error

    def eos(self, reorderlevel=0, acc=False):
        """ Return the expected order size.
        The reorder level is only required for the lost sales situation.
        """
        if self.dobr_error == 0:
            reorderlevel = self.adjust_rol(reorderlevel)
            sf, val_acc = self._supply_factor(reorderlevel, acc=True)
            eol = self._eol(reorderlevel)
            if eol > EPS:
                eos = sf*self.mean*self.reviewperiod/eol
            else:
                eos = self._max_oq()
            if acc:
                return eos, val_acc
            return eos
        return self.dobr_error

    def supply(self, reorderlevel=0, acc=False):
        """Return the expected supply per review period.
        The reorder level is only required for the lost sales situation.
        """
        if self.dobr_error == 0:
            reorderlevel = self.adjust_rol(reorderlevel)
            sf, val_acc = self._supply_factor(reorderlevel, acc=True)
            if acc:
                return sf*self.reviewperiod*self.mean, val_acc
            return sf*self.reviewperiod*self.mean
        return self.dobr_error

    def est(self, reorderlevel, acc=False):
        """ Return the expected sojourn time (time between delivery and sales)
        of the items in the inventory.
        """
        if self.dobr_error == 0:
            reorderlevel = self.adjust_rol(reorderlevel)
            est = 0.
            eioh_l, val_acc = self._eioh("L", reorderlevel, acc=True)
            eioh_rl = self._eioh("RL", reorderlevel)
            fr = max(0., (eioh_l - eioh_rl)/(self.reviewperiod*self.mean))
            if fr > EPS:
                eioh_c = self._eioh_c(reorderlevel, eioh_l)
                if self.shelflife > 0:
                    ez = self._ez(reorderlevel)
                    fresh = (self.shelflife - eioh_c/((ez + fr)*self.mean))*(1+ez/fr)
                    fresh = min(self.shelflife - (self.reviewperiod-1)/2,
                                max((self.reviewperiod+1)/2, fresh))
                    est = self.shelflife - fresh
                else:
                    # Non perishable
                    est = eioh_c/(fr*self.mean)
                if self.reviewperiod > 2 or self.shelflife > 0:
                    val_acc = VAL_APPROX
            if acc:
                return max(0, est), val_acc
            return max(0., est)
        return self.dobr_error

    def ew(self, reorderlevel, acc=False):
        """Return the expected outdating."""
        if self.dobr_error == 0:
            if self.shelflife > 0:
                reorderlevel = self.adjust_rol(reorderlevel)
                ez, val_acc = self._ez(reorderlevel, acc=True)
                ew = ez*self.reviewperiod*self.mean
                if acc:
                    return ew, val_acc
                return ew
            if acc:
                return 0., VAL_EXACT
            return 0.
        return self.dobr_error

    def eua(self, reorderlevel=0, unitcap=0., acc=False):
        """Return the expected number of unit arrivals per review period.

        Keyword arguments:
        reorderlevel -- the reorder level (default 0)
        unitcap      -- the capacity (default 0)
        """
        if self.dobr_error == 0 and self._val_unitcap(unitcap):
            reorderlevel = self.adjust_rol(reorderlevel)
            return self._eua(reorderlevel, unitcap, acc=acc)
        return self.dobr_error

    def enb(self, reorderlevel=0, acc=False):
        """Return the expected number of batches per review period.

        Keyword arguments:
        reorderlevel -- the reorder level (default 0)
        """
        if self.dobr_error == 0:
            reorderlevel = self.adjust_rol(reorderlevel)
            return self._enb(reorderlevel, acc=acc)
        return self.dobr_error

    def eusl_l(self, reorderlevel=0, unitcap=0., acc=False):
        """Return the expected number of locations per review period.

        Keyword arguments:
        reorderlevel -- the reorder level (default 0)
        unitcap      -- the capacity (default 0)
        """
        if self.dobr_error == 0 and self._val_unitcap(unitcap):
            reorderlevel = self.adjust_rol(reorderlevel)
            eua, val_acc = self._eua(reorderlevel, unitcap, acc=acc)
            eos, val_acc = self.eos(reorderlevel=reorderlevel, acc=acc)
            eusl, val_acc = self._eusl(reorderlevel, unitcap, eos, acc=acc)
            return (eusl + eua), val_acc
        return self.dobr_error

    def poc(self, reorderlevel, capacity=0, acc=False):
        """ Return the probability of overflow of the shelf capacity
        after the lead-time for discrete demand.

        Keyword arguments:
        capacity -- the capacity (default 0)
        """
        if self.dobr_error == 0 and self._val_shelfspace(capacity):
            reorderlevel = self.adjust_rol(reorderlevel)
            capacity = self._adjust_shelfspace(capacity)
            if capacity > EPS:
                return self._poc("L", reorderlevel,
                                 capacity=capacity, acc=acc)
            if acc:
                return 0., VAL_EXACT
            return 0.
        return self.dobr_error

    def eibr(self, reorderlevel, capacity=0, acc=False):
        """ Return the expected inventory above capacity (shelf: in the backroom)
        after the lead-time for discrete demand.

        Keyword arguments:
        capacity -- the capacity (default 0)
        """
        if self.dobr_error == 0 and self._val_shelfspace(capacity):
            reorderlevel = self.adjust_rol(reorderlevel)
            capacity = self._adjust_shelfspace(capacity)
            if capacity > EPS:
                return self._eioh("L", reorderlevel, capacity=capacity, acc=acc)
            if acc:
                return 0., VAL_EXACT
            return 0.
        return self.dobr_error

    def enir(self, reorderlevel, capacity=0, acc=False):
        """ Return the expected number of idle replenishments
        from the backroom.

        Keyword arguments:
        capacity -- the capacity (default 0)
        """
        if self.dobr_error == 0 and self._val_shelfspace(capacity):
            reorderlevel = self.adjust_rol(reorderlevel)
            capacity = self._adjust_shelfspace(capacity)
            if capacity > EPS:
                enir = 0.
                poc_l , val_acc = self._poc("L", reorderlevel,
                                            capacity=capacity, acc=True)
                encr = self._prob_pos_demand(capacity)*poc_l
                if encr < 0.001:
                    enir = self._prob_pos_demand(0)*poc_l
                else:
                    poc_rl = self._poc("RL", reorderlevel, capacity=capacity)
                    enir = (poc_l*(self._prob_pos_demand(capacity)
                                   -self._prob_pos_demand(0))
                            + poc_rl*encr)
                # Return only positive values
                enir = max(0., enir)
                if acc:
                    return enir, val_acc
                return enir
            if acc:
                return 0., VAL_EXACT
            return 0.
        return self.dobr_error

    def encr(self, reorderlevel, capacity=0, acc=False):
        """ Return the expected number of concurrent replenishments
        from the backroom.

        Keyword arguments:
        capacity -- the capacity (default 0)
        """
        if self.dobr_error == 0 and self._val_shelfspace(capacity):
            reorderlevel = self.adjust_rol(reorderlevel)
            capacity = self._adjust_shelfspace(capacity)
            if capacity > EPS:
                return self._encr(reorderlevel, capacity, acc=acc)
            if acc:
                return 0., VAL_EXACT
            return 0.
        return self.dobr_error

    def targetfillrate(self, target_fr,
                       xguess=None, yguess=None, xtol=0.001, acc=False):
        """ Return the minimum reorder level that gives the target fill rate.

        Keyword arguments:
        xguess -- Initial guess for the reorder level
        yguess -- Value for the fill rate at the initial guess
        xtol   -- Absolute interval for the returned reorder level (default = 0.001)
        """
        if (self.dobr_error == 0 and self._val_target(target_fr)):
            rol = self._targetrate(self._root_fr, target_fr,
                                   xguess, yguess, xtol)
            if acc:
                return rol, self.accuracy()
            return rol
        return self.dobr_error

    def targetreadyrate(self, target_rr,
                        xguess=1., yguess=-1., xtol=0.001, acc=False):
        """ Return the minimum reorder level that gives the target
        discrete ready rate.

        Keyword arguments:
        xguess -- Initial guess for the reorder level
        yguess -- Value for the ready rate at the initial guess
        xtol   -- Absolute interval for the returned reorder level (default = 0.001)
        """
        if (self.dobr_error == 0 and self._val_target(target_rr)):
            rol = self._targetrate(self._root_rr, target_rr,
                                   xguess, yguess, xtol)
            if acc:
                return rol, self.accuracy()
            return rol
        return self.dobr_error

    def targetservdiffrate(self, target_sdr,
                           xguess=1., yguess=-1., xtol=0.001, acc=False):
        """ Return the minimum reorder level that gives the target
        service differentiation rate.

        Keyword arguments:
        xguess -- Initial guess for the reorder level
        yguess -- Value for the service differentiation rate at the initial guess
        xtol   -- Absolute interval for the returned reorder level (default = 0.001)
        """
        if (self.dobr_error == 0 and self._val_target(target_sdr)):
            rol = self._targetrate(self._root_sdr, target_sdr,
                                   xguess, yguess, xtol)
            if acc:
                return rol, self.accuracy()
            return rol
        return self.dobr_error

    def mincost(self, cost_function, xlow=1, xtol=0.001):
        """ Return the reorder level that minimizes the cost function.

        Keyword arguments:
        xlow -- Initial guess for the reorder level
        xtol -- Absolute interval for the returned reorder level (default = 0.001)
        """
        if self.dobr_error == 0:
            rol = self._mincost(cost_function, xlow, xtol)
            return rol
        return self.dobr_error

    def dist_name(self):
        """Return the name of the fitted distribution."""
        return "Unknown"

    @staticmethod
    def order_situation():
        """Return the default response to out-of-stocks."""
        return "back ordering"

    def _root_fr(self, reorderlevel, target_fr):
        # Return the fill rate minus the target.
        return self._fr(reorderlevel) - target_fr

    def _root_rr(self, reorderlevel, target_rr):
        # Return the discrete ready rate minus the target.
        # Discrete ready rate = P[D_RL>s]
        rr = self._poc("RL", reorderlevel)
        return rr - target_rr

    def _root_sdr(self, reorderlevel, target_sdr):
        # Return the ratio of poc's minus the target.
        sdr = max(0., self._poc("RL", reorderlevel)
                  /self._poc("L", reorderlevel))
        return sdr - target_sdr

    def _targetrate(self, rate, target, xguess, yguess, xtol):
        # Find the reorder level that satisfies the target
        # Start with bracketing
        if xguess is None:
            # Initial guess: 1.5 times mean R+L
            test_rol = self.adjust_rol(max(1.,
                1.5*(self.reviewperiod+self.leadtime)*self.mean))
            test_rate = rate(test_rol, target)
        else:
            # Initial guess is passed
            test_rol = xguess
            if yguess is None:
                test_rate = rate(test_rol, target)
            else:
                # Value of rate at initial guess is passed
                test_rate = yguess - target
        if test_rate < 0.:
            # Bottom = test
            low_rol = test_rol
            low_rate = test_rate
            high_rol = self.adjust_rol(-self.ioq)
            high_rate = -1.
            while high_rate < 0.:
                # Determine next test
                if (high_rate < 0.) or abs(high_rate-low_rate) < EPS:
                    test_rol = self.adjust_rol(max(1.,test_rol*2))
                else:
                    # Extrapolation for new test value
                    test_rol = self.adjust_rol(high_rol
                               + max(1, (1.-high_rate)*(high_rol-low_rol)
                                     /(high_rate-low_rate)))

                test_rate = rate(test_rol, target)
                if -1. < high_rate < 0.:
                    # We have an upper bracket, but not yet above the target
                    # bottom = high
                    low_rol = high_rol
                    low_rate = high_rate
                # High = test
                high_rol = test_rol
                high_rate = test_rate
        else:
            # High = mid
            high_rol = test_rol
            high_rate = test_rate
            # Bottom: -ioq
            low_rol = self.adjust_rol(-self.ioq)
            low_rate = 0.

        if self.discrete:
            # Basic bisection: integer reorder level required
            while high_rol - low_rol > 1:
                test_rol = low_rol + int((high_rol-low_rol)/2)
                test_rate = rate(test_rol, target)
                if test_rate < 0.:
                    low_rol = test_rol
                    low_rate = test_rate
                else:
                    high_rol = test_rol
                    high_rate = test_rate
        else:
            # Bisection using brentq
            high_rol = brentq(rate, low_rol, high_rol,
                              xtol=xtol, args=target)

        return high_rol

    def _mincost(self, cost_function, xlow, xtol):
        # Find the reorder level that minimizes the cost function
        # using golden ratio search

        # Value golden ratio
        gold = 0.618
        # Start with bracketing
        steps_b = 0
        a = xlow
        f_a = cost_function(self, a)
        b = a + self.reviewperiod*self.mean
        if self.discrete:
            b = math.ceil(b)
        f_b = cost_function(self, b)
        if f_a > f_b:
            c = b + (1 + gold)*(b - a)
            if self.discrete:
                c = math.ceil(c)
            f_c = cost_function(self, c)
            while f_b > f_c:
                steps_b += 1
                a = b
                f_a = f_b
                b = c
                f_b = f_c
                c = b + (1 + gold)*(b - a)
                if self.discrete:
                    c = math.ceil(c)
                f_c = cost_function(self, c)
                # print(steps_b, a, b, c)
                # print(steps_b, f_a, f_b, f_c)
            b = c
            f_b = f_c

        # Basic bisection: integer reorder level required
        # steps = 0
        if self.discrete:
            xtol = 1
        while b - a > xtol:
            # steps += 1
            l = b - gold*(b - a)
            if self.discrete:
                l = max(a+1, int(l))
            f_l = cost_function(self, l)
            r = a + gold*(b - a)
            if self.discrete:
                r = min(b-1, int(r))
            f_r = cost_function(self, r)
            if f_l < f_r:
                # Left interval
                b = r
                r = l
                f_r = f_l
                l = b - gold*(b - a)
                if self.discrete:
                    l = max(a+1, int(l))
                f_l = cost_function(self, l)
            else:
                # Right interval
                a = l
                l = r
                f_l = f_r
                r = a + gold*(b - a)
                if self.discrete:
                    r = min(b-1, int(r))
                f_r = cost_function(self, r)

        # print(steps_b, steps)
        return r

    def calc_kpis(self, kpi_list, reorderlevel,
                  unitcap=0, capacity=0):
        """ Return the value of the KPIs in the passed list.

        Keyword arguments:
        unitcap  -- the unit load capacity (default = 0.)
        capacity -- the capacity (default 0)
        """
        # Determine the base KPIs
        eioh_l, b_acc = self._eioh("L", reorderlevel, acc=True)
        eioh_rl = self._eioh("RL", reorderlevel)

        kpi_results = {}
        for kpi in kpi_list:
            # Reset to NOT APPLICABLE
            kpi_results[kpi] = [0., VAL_NOTAPPLIC]

           # Only calculate if requested
            if kpi == "Fillrate":
                fr = self._fr(reorderlevel, eioh_l=eioh_l, eioh_rl=eioh_rl)
                kpi_results[kpi] = [fr, b_acc]
                # kpi_results[kpi] = self.fillrate(reorderlevel, acc=True)
            elif kpi == "Readyrate":
                kpi_results[kpi] = self.readyrate(reorderlevel, acc=True)
            elif kpi == "ELT" and self.var_leadtime > EPS:
                kpi_results[kpi] = [self.leadtime, VAL_EXACT]
            elif kpi == "EBO_L" and (not self.lost_sales):
                kpi_results[kpi] = self._ebo("L", reorderlevel,
                                             eioh=eioh_l, acc=True)
            elif kpi == "EBO_RL" and (not self.lost_sales):
                kpi_results[kpi] = self._ebo("RL", reorderlevel,
                                             eioh=eioh_rl, acc=True)
            elif kpi == "EIOH_L":
                kpi_results[kpi] = [eioh_l, b_acc]
                # kpi_results[kpi] = self.eioh_l(reorderlevel, acc=True)
            elif kpi == "EIOH_RL":
                kpi_results[kpi] = [eioh_rl, b_acc]
                # kpi_results[kpi] = self.eioh_rl(reorderlevel, acc=True)
            elif kpi == "EOL":
                kpi_results[kpi] = self.eol(reorderlevel=reorderlevel, acc=True)
            elif kpi == "EOS":
                kpi_results[kpi] = self.eos(reorderlevel=reorderlevel, acc=True)
            elif kpi == "ESUP":
                kpi_results[kpi] = self.supply(reorderlevel=reorderlevel, acc=True)
            elif kpi == "EUA":
                kpi_results[kpi] = self.eua(reorderlevel=reorderlevel,
                                          unitcap=unitcap, acc=True)
            elif kpi == "ENB":
                kpi_results[kpi] = self.enb(reorderlevel=reorderlevel, acc=True)
            elif kpi == "EUSL_L":
                kpi_results[kpi] = self.eusl_l(reorderlevel=reorderlevel,
                                          unitcap=unitcap, acc=True)
            elif kpi == "EW" and self.shelflife > 0:
                kpi_results[kpi] = self.ew(reorderlevel, acc=True)
            elif kpi == "EST":
                kpi_results[kpi] = self.est(reorderlevel, acc=True)
            elif kpi == "POC" and capacity > 0:
                kpi_results[kpi] = self.poc(reorderlevel,
                                         capacity=capacity, acc=True)
            elif kpi == "EIBR" and capacity > 0:
                kpi_results[kpi] = self.eibr(reorderlevel,
                                          capacity=capacity, acc=True)
            elif kpi == "ENIR" and capacity > 0:
                kpi_results[kpi] = self.enir(reorderlevel,
                                          capacity=capacity, acc=True)
            elif kpi == "ENCR" and capacity > 0:
                kpi_results[kpi] = self.encr(reorderlevel,
                                         capacity=capacity, acc=True)

        return kpi_results

    def print_header(self, reorderlevel, unitcap, capacity):
        """Print the parameters of the inventory system."""

        print(" KPI's for a periodic review inventory system"
              + f" assuming {self.order_situation()}")
        print(f"  and {self.dist_name()} distributed demand"
              + f" (Mean={round(self.mean, 3):g}, "
              + f"StDev={round(math.sqrt(self.vtm*self.mean),3):g}),")
        line = f"  L={round(self.leadtime,3):g}"
        if self.stdev_leadtime > EPS:
            line = line + f", StDevL={round(self.stdev_leadtime,3):g},"
        else:
            line = line + " (fixed),"
        line = line + f" R={round(self.reviewperiod,3):g},"
        line = line + f" IOQ={self.ioq},"
        if self.moq > self.ioq:
            line = line + f" MOQ={self.moq},"
        if len(line) > 65:
            print(line)
            line = " "
        if unitcap > EPS:
            line = line + f" U={unitcap},"
        if len(line) > 65:
            print(line)
            line = " "
        if capacity > EPS:
            line = line + f" V={capacity},"
        if len(line) > 65:
            print(line)
            line = " "
        if self.shelflife > 0:
            line = line + f" m={self.shelflife},"
        if len(line) > 65:
            print(line)
            line = " "
        line = line + f" and s={reorderlevel}"
        print(line)
        print(" ".ljust(70, "-")+" ")

    def print_kpis(self, reorderlevel, unitcap=0, capacity=0):
        """Print the basic KPI's of the inventory system.

        Keyword arguments:
        unitcap  -- the unit load capacity (default = 0.)
        capacity -- the capacity (default 0)
        """
        if self.dobr_error == 0:
            rol = self.adjust_rol(reorderlevel)

            # Convert to data_dict
            data_dict = dict(DEFAULT_DATA)
            data_dict["distribution"] = self.dist_name()
            if self.order_situation() == "lost sales":
                data_dict["lostsales"] = True
            else:
                data_dict["lostsales"] = False
            data_dict["leadtime"] = self.leadtime
            data_dict["stdev_leadtime"] = self.stdev_leadtime
            data_dict["reviewperiod"] = self.reviewperiod
            data_dict["mean_perioddemand"] = self.mean
            data_dict["stdev_perioddemand"] = self.variance**0.5
            data_dict["ioq"] = self.ioq
            data_dict["moq"] = self.moq
            data_dict["reorderlevel"] = rol
            data_dict["unitcap"] = unitcap
            data_dict["shelfspace"] = capacity
            data_dict["shelflife"] = self.shelflife

            kpi_results = self.calc_kpis(KPI_CATALOGUE, rol,
                                         unitcap=unitcap, capacity=capacity)
            print_kpis_ana(kpi_results, data_dict)
        else:
            print("DoBr: Not a valid data set.")


class InvSysGammaBO(InvSys):
    """ A class used to describe (R,s,nQ) inventory systems
    with Gamma distributed demand."""

    def __init__(self, mean_perioddemand, stdev_perioddemand, leadtime,
                 stdev_leadtime=0., reviewperiod=1, ioq=1, printerror=True):
        """ Initialize class variables:
        mean_perioddemand  -- mean demand per period (> 0)
        stdev_perioddemand -- standard deviation of the demand per period (> 0)
        leadtime           -- time between ordering and delivery in periods

        Keyword arguments:
        stdev_leadtime     -- the standard deviation of the leadtime (default 0.0)
        reviewperiod -- the time between two reviews (default 1.0)
        ioq          -- the incremental order quantity (default 1)
        printerror   -- print error messages to the console (default True)
        """

        # Call abstract parent class: note that moq=0
        super().__init__(
            mean_perioddemand, stdev_perioddemand,
            leadtime, stdev_leadtime=stdev_leadtime, reviewperiod=reviewperiod,
            ioq=ioq, moq=0)

        # Adjust ioq and moq to floats
        self.ioq = float(self.ioq)
        self.moq = float(self.moq)

        if self.dobr_error != 0 and printerror:
            print("DoBr: " + ERROR_CODES[self.dobr_error])

    def dist_name(self):
        """Return the name of the fitted distribution."""
        return "Gamma"

    def accuracy(self):
        """Generic accuracy of the KPIs in this class, based on lead-time
        uncertainty."""
        if self.leadtime > EPS and self.var_leadtime > EPS:
            return VAL_APPROX
        return VAL_EXACT

    def _supply_factor(self, reorderlevel, acc=False):
        # No loss with backordering
        if acc:
            return 1., VAL_EXACT
        return 1.

    def _prob_pos_demand(self, capacity):
        # Return probability positive demand above capacity
        theta = self.vtm
        alpha = self.reviewperiod*self.mean/theta
        return 1. - gamma.cdf(capacity, alpha, scale=theta)

    def _eioh(self, cycle, reorderlevel, capacity=0, acc=False):
        # Expected inventory on-hand for a given cycle length
        # for Gamma distributed demand.
        s = reorderlevel - capacity
        q = self.ioq
        if cycle == "L" and self.leadtime < EPS:
            # ioh_l is equal to ip+
            if s+q > EPS:
                # Positive inventory
                if q > EPS:
                    # Uniform distributed ip+
                    max_ioh = s+q
                    min_ioh = max(0., s)
                    eioh = ((max_ioh-min_ioh)/q)*(max_ioh+min_ioh)/2.
                else:
                    eioh = s
            else:
                eioh = 0.
        else:
            theta = self.vtm + self._vtmplus(cycle)
            mu = self._mu_cycle(cycle)
            alpha = mu/theta

            if q > EPS:
                # (R,s,nQ) system
                eioh = (((alpha+1)*alpha*theta*theta/(2*q))
                        *(gamma.cdf(max(0., s+q), alpha+2, scale=theta)
                          -gamma.cdf(max(0., s), alpha+2, scale=theta))
                        -((s+q)/q)*alpha*theta*gamma.cdf(max(0., s+q), alpha+1, scale=theta)
                        +(s*alpha*theta/q)*gamma.cdf(max(0., s), alpha+1, scale=theta)
                        +((s+q)**2/(2*q))*gamma.cdf(max(0., s+q), alpha, scale=theta)
                        -(s*s/(2*q))*gamma.cdf(max(0., s), alpha, scale=theta))
            else:
                # (R,S) system
                eioh = (s*gamma.cdf(max(0., s), alpha, scale=theta)
                        -mu*gamma.cdf(max(0., s), alpha+1, scale=theta))
        # Return only positive values
        eioh = max(0., eioh)
        if acc:
            return eioh, self.accuracy()
        return eioh

    def _ebo(self, cycle, reorderlevel, eioh=None, acc=False):
        # Expected backorders based on the E[IOH] for the given cycle length
        # for Gamma distributed demand.
        if eioh is None:
            eioh = self._eioh(cycle, reorderlevel)
        ebo = max(0., eioh + (self._mu_cycle(cycle) - (reorderlevel+self.ioq/2)))
        if acc:
            return ebo, self.accuracy()
        return ebo

    def _poc(self, cycle, reorderlevel, capacity=0, acc=False):
        # Probability of on-hand inventory above the capacity
        # for Gamma distributed demand.
        s = reorderlevel - capacity
        q = self.ioq
        theta = self.vtm + self._vtmplus(cycle)
        mu = self._mu_cycle(cycle)
        alpha = mu/theta

        if q > EPS:
            # (R,s,nQ) system
            poc = (((s+q)/q)*gamma.cdf(max(0., s+q), alpha, scale=theta)
                     -(s/q)*gamma.cdf(max(0., s), alpha, scale=theta)
                     -(alpha*theta/q)
                      *(gamma.cdf(max(0., s+q), alpha+1, scale=theta)
                        -gamma.cdf(max(0., s), alpha+1, scale=theta)))
        else:
            # (R,S) system
            poc = gamma.cdf(max(0., s), alpha, scale=theta)
        # Return only positive values
        poc = max (0., poc)
        if acc:
            return poc, self.accuracy()
        return poc

    def _eol(self, reorderlevel, acc=False):
        # Return the expected number of order lines per review period
        # for Gamma distributed demand.
        q = self.ioq
        mu = self.reviewperiod*self.mean

        if q > EPS:
            # (R,s,nQ) system
            theta = self.vtm
            alpha = mu/theta
            eol = (1.-gamma.cdf(q, alpha, scale=theta)
                       +(alpha*theta/q)*gamma.cdf(q, alpha+1, scale=theta))
        else:
            # (R,S) system
            eol = 1.
        # Return only positive values
        eol = max(0., eol)
        if acc:
            return eol, VAL_EXACT
        return eol

    def _ez(self, reorderlevel, acc=False):
        # Expected relative outdating z not known yet
        # for Gamma distributed demand.
        if acc:
            return 0., VAL_NOTYET
        return 0.

    def _eua(self, reorderlevel, unitcap, acc=False):
        # Return the expected unit arrivals per review period
        # for Gamma distributed demand.
        q = max(1.0, self.ioq)
        cap = max(q, float(unitcap))
        mu = self.reviewperiod*self.mean

        theta = self.vtm
        alpha = mu/theta
        # (R,s,nQ) system
        eua = 0.
        inc_eua = 1.
        n = 1
        cdf_a_min = gamma.cdf(0, alpha, scale=theta)
        cdf_a_curr = gamma.cdf(q, alpha, scale=theta)
        cdf_a_plus = gamma.cdf(2*q, alpha, scale=theta)
        cdf_a1_min = gamma.cdf(0, alpha+1, scale=theta)
        cdf_a1_curr = gamma.cdf(q, alpha+1, scale=theta)
        cdf_a1_plus = gamma.cdf(2*q, alpha+1, scale=theta)
        while n == 1 or cdf_a_curr < 1.0-EPS:
            inc_eua = ((n+1)*q*cdf_a_plus - 2*n*q*cdf_a_curr + (n-1)*q*cdf_a_min
                           + mu*(2*cdf_a1_curr - cdf_a1_min - cdf_a1_plus))
            eua += inc_eua*math.ceil(n*q/cap)
            n += 1
            # Shift
            cdf_a_min = cdf_a_curr
            cdf_a1_min = cdf_a1_curr
            cdf_a_curr = cdf_a_plus
            cdf_a1_curr = cdf_a1_plus
            cdf_a_plus = gamma.cdf((n+1)*q, alpha, scale=theta)
            cdf_a1_plus = gamma.cdf((n+1)*q, alpha+1, scale=theta)
        eua /= q
        # Return only positive etu values
        eua = max(0., eua)
        if acc and self.ioq < 1.:
            return eua, VAL_APPROX
        if acc:
            return eua, VAL_EXACT
        return eua

    def _enb(self, reorderlevel, acc=False):
        # Expected number of batches not known yet
        # for Gamma distributed demand.
        if acc:
            return 0., VAL_NOTYET
        return 0.

    def _eusl(self, reorderlevel, unitcap, eos, acc=False):
        # Expected number of locations not known yet
        # for Gamma distributed demand.
        if acc:
            return 0., VAL_NOTYET
        return 0.

    def _encr(self, reorderlevel, capacity, acc=False):
        poc_l, val_acc = self._poc("L", reorderlevel,
                                   capacity=capacity, acc=True)
        # Correct for demand during review period
        encr = self._prob_pos_demand(capacity)*poc_l
        # Return only positive values
        encr = max(0., encr)
        if acc:
            return encr, val_acc
        return encr


class InvSysNormalBO(InvSys):
    """ A class used to describe (R,s,nQ) inventory systems
    with Normal distributed demand."""

    def __init__(self, mean_perioddemand, stdev_perioddemand, leadtime,
                 stdev_leadtime=0., reviewperiod=1, ioq=1, printerror=True):
        """ Initialize class variables:
        mean_perioddemand  -- mean demand per period (> 0)
        stdev_perioddemand -- standard deviation of the demand per period (> 0)
        leadtime           -- time between ordering and delivery in periods

        Keyword arguments:
        stdev_leadtime     -- the standard deviation of the leadtime (default 0.0)
        reviewperiod -- the time between two reviews (default 1.0)
        ioq          -- the incremental order quantity (default 1)
        printerror   -- print error messages to the console (default True)
        """

        # Call abstract parent class: note that moq=0
        super().__init__(
            mean_perioddemand, stdev_perioddemand,
            leadtime, stdev_leadtime=stdev_leadtime, reviewperiod=reviewperiod,
            ioq=ioq, moq=0)

        if self.dobr_error != 0 and printerror:
            print("DoBr: " + ERROR_CODES[self.dobr_error])

    def dist_name(self):
        """Return the name of the fitted distribution."""
        return "Normal"

    def accuracy(self):
        """Generic accuracy of the KPIs in this class, based on lead-time
        uncertainty."""
        if self.leadtime > EPS and self.var_leadtime > EPS:
            return VAL_APPROX
        return VAL_EXACT

    def _supply_factor(self, reorderlevel, acc=False):
        # No loss with backordering
        if acc:
            return 1., VAL_EXACT
        return 1.

    def _prob_pos_demand(self, capacity):
        # Return probability positive demand above capacity
        # for Normal distributed demand.
        mu = self.reviewperiod*self.mean
        sigma = math.sqrt(self.vtm*mu)
        k = (capacity-mu)/sigma
        return 1.-norm.cdf(k)

    def _eioh(self, cycle, reorderlevel, capacity=0, acc=False):
        # Expected inventory on-hand for a given cycle length
        # for Normal distributed demand.
        s = reorderlevel - capacity
        q = self.ioq
        if cycle == "L" and self.leadtime < EPS:
            # ioh_l is equal to ip+
            if s+q > EPS:
                # Positive inventory
                if q > EPS:
                    # Uniform distributed ip+
                    max_ioh = s+q
                    min_ioh = max(0., s)
                    eioh = ((max_ioh-min_ioh)/q)*(max_ioh+min_ioh)/2.
                else:
                    eioh = s
            else:
                eioh = 0.
        else:
            theta = self.vtm + self._vtmplus(cycle)
            mu = self._mu_cycle(cycle)
            sigma = math.sqrt(theta*mu)

            k = (s-mu)/sigma
            c = q/sigma
            if q > EPS:
                # (R,s,nQ) system
                eioh = ((k/c+1)*(sigma/2)*norm.pdf(k+c)
                        -(k*sigma/(2*c))*norm.pdf(k)
                        +((k*k+1)*(sigma/(2*c))
                          +(k+c)*sigma-c*sigma/2)*norm.cdf(k+c)
                        -(k*k+1)*(sigma/(2*c))*norm.cdf(k))
            else:
                # (R,S) system
                eioh = sigma*norm.pdf(k)-(mu-s)*norm.cdf(k)
        # Return only positive values
        eioh = max(0., eioh)
        if acc:
            return eioh, self.accuracy()
        return eioh

    def _ebo(self, cycle, reorderlevel, eioh=None, acc=False):
        # Expected backorders based on the E[IOH] for the given cycle length
        # for Normal distributed demand.
        if eioh is None:
            eioh = self._eioh(cycle, reorderlevel)
        ebo = max(0., eioh + (self._mu_cycle(cycle) - (reorderlevel+self.ioq/2)))
        if acc:
            return ebo, self.accuracy()
        return ebo

    def _poc(self, cycle, reorderlevel, capacity=0, acc=False):
        # Probability of on-hand inventory above the capacity
        # for Normal distributed demand.
        s = reorderlevel - capacity
        q = self.ioq
        theta = self.vtm + self._vtmplus(cycle)
        mu = self._mu_cycle(cycle)
        sigma = math.sqrt(theta*mu)

        k = (s-mu)/sigma
        if q > 0:
            # (R,s,nQ) system
            c = q/sigma
            poc = (((s+q)/q)*norm.cdf(k+c)-(s/q)*norm.cdf(k)
                      +(1./c)*(norm.pdf(k+c)-norm.pdf(k))
                      -(mu/q)*(norm.cdf(k+c)-norm.cdf(k)))
        else:
            # (R,S) system
            poc = norm.cdf(k)
        # Return only positive values
        poc = max(0., poc)
        if acc:
            return poc, self.accuracy()
        return poc

    def _eol(self, reorderlevel, acc=False):
        # Return the expected number of order lines per review period
        # for Normal distributed demand.
        q = self.ioq
        mu = self.reviewperiod*self.mean
        sigma = math.sqrt(self.vtm*mu)

        j = mu/sigma
        if q > EPS:
            # (R,s,nQ) system
            c = q/sigma
            eol = (1.-norm.cdf(c-j)
                       +(1.0/c)*(norm.pdf(-j)-norm.pdf(c-j))
                       +(mu/q)*(norm.cdf(c-j)-norm.cdf(-j)))
        else:
            # (R,S) system
            eol = 1.-norm.cdf(-j)
        # Check on EOS
        if mu/eol < q:
            eol = mu/q
        # Return only positive eol values
        eol = max(0., eol)
        if acc:
            return eol, VAL_EXACT
        return eol

    def _ez(self, reorderlevel, acc=False):
        # Expected relative outdating z not known yet
        # for Normal distributed demand.
        if acc:
            return 0., VAL_NOTYET
        return 0.

    def _eua(self, reorderlevel, unitcap, acc=False):
        # Return the expected unit arrivals per review period
        # for Normal distributed demand.
        q = max(1.0, self.ioq)
        cap = max(q, float(unitcap))
        mu = self.reviewperiod*self.mean
        sigma = math.sqrt(self.vtm*mu)

        j = mu/sigma
        c = q/sigma
        # (R,s,nQ) system
        eua = 0.
        inc_eua = 1.
        n = 1
        pdf_min = norm.pdf(-j)
        pdf_curr = norm.pdf(c-j)
        pdf_plus = norm.pdf(2*c-j)
        cdf_min = norm.cdf(-j)
        cdf_curr = norm.cdf(c-j)
        cdf_plus = norm.cdf(2*c-j)
        while n == 1 or cdf_curr < 1.0-EPS:
            inc_eua = (sigma*(pdf_min - 2*pdf_curr + pdf_plus)
                       + ((n+1)*q-mu)*cdf_plus - 2*(n*q-mu)*cdf_curr
                       + ((n-1)*q-mu)*cdf_min)
            eua += inc_eua*math.ceil(n*q/cap)
            n += 1
            # Shift
            pdf_min = pdf_curr
            cdf_min = cdf_curr
            pdf_curr = pdf_plus
            cdf_curr = cdf_plus
            pdf_plus = norm.pdf((n+1)*c-j)
            cdf_plus = norm.cdf((n+1)*c-j)
        eua /= q
        # Return only positive etu values
        eua = max(0., eua)
        if acc and self.ioq < 1.:
            return eua, VAL_APPROX
        if acc:
            return eua, VAL_EXACT
        return eua

    def _enb(self, reorderlevel, acc=False):
        # Expected number of batches not known yet
        # for Normal distributed demand.
        if acc:
            return 0., VAL_NOTYET
        return 0.

    def _eusl(self, reorderlevel, unitcap, eos, acc=False):
        # Expected number of locations not known yet
        # for Normal distributed demand.
        if acc:
            return 0., VAL_NOTYET
        return 0.

    def _encr(self, reorderlevel, capacity, acc=False):
        poc_l, val_acc = self._poc("L", reorderlevel,
                                   capacity=capacity, acc=True)
        # Correct for demand during review period
        encr = self._prob_pos_demand(capacity)*poc_l
        # Return only positive values
        encr = max(0., encr)
        if acc:
            return encr, val_acc
        return encr


class InvSysDiscreteBO(InvSys):
    """ A class used to describe (R,s,nQ,S) inventory systems.
    The methods of this subclass assume (mixtures of) discrete demand
    distributions, backordering, and only the reorder level
    as parameter.
    """

    def __init__(
            self, mean_perioddemand, stdev_perioddemand, leadtime,
            stdev_leadtime=0., reviewperiod=1,
            ioq=1, moq=0, shelflife=0,
            printerror=True, empiricalpmf=None, usecp=False):
        """ Initialize class variables:
        mean_perioddemand  -- mean demand per period (> 0)
        stdev_perioddemand -- standard deviation of the demand per period (> 0)
        leadtime           -- time between ordering and delivery in periods

        Keyword arguments:
        stdev_leadtime     -- the standard deviation of the leadtime (default 0.0)
        reviewperiod -- the time between two reviews (default 1.0)
        ioq          -- the incremental order quantity (default 1)
        moq          -- the minimal order quantity (default 0)
        shelflife    -- the remaining shelflife upon entering the stock point
        printerror   -- print error messages to the console (default True)
        empiricalpmf -- the empirical pmf for the base period (default None)
        usecp        -- use Compound Poisson if VTM>1 (default False)
        """
        # Is an empirical discrete distribution given for the base period?
        sub_error = 0
        if not empiricalpmf is None:
            self.base = dobr_dist.EmpiricalDistArray(empiricalpmf)
            if self.base.dist_error == 0:
                # Override passed mean and stdev by the values derived
                # from the empirical data
                mean_perioddemand = self.base.mean
                stdev_perioddemand = math.sqrt(self.base.vtm*self.base.mean)
            else:
                sub_error = self.base.dist_error

        # Call parent class init method
        super().__init__(
            mean_perioddemand, stdev_perioddemand, leadtime,
            stdev_leadtime=stdev_leadtime, reviewperiod=reviewperiod,
            ioq=ioq, moq=moq, shelflife=shelflife,
            sub_error=sub_error)

        # Check if ioq and moq are discrete
        self._val_oq()

        if self.dobr_error == 0:
            # Adjust ioq and moq to integers
            self.ioq = int(round(self.ioq))
            self.moq = max(int(round(self.moq)), self.ioq)

            # First, fit the base distribution
            if empiricalpmf is None:
                self.base = dobr_dist.select_dist(
                    self.mean, self.vtm, trycp=usecp)
                if self.base is None:
                    self.dobr_error = -9903
            if self.dobr_error == 0:
                # Which demand distribution cycles are required?
                self._list_dists()
                # Generate the required distributions
                self._create_dists(usecp)
                # Inventory distributions
                self.inv_dist = {"IP+": None, "IP-": None,
                                 "IOH_L": None, "IOH_RL": None}
                # Create ip+ distribution (independent of reorder level)
                self.inv_dist["IP+"] = dobr_dist.EmpiricalDistArray(
                    self._create_dist_ip_plus())
                # No reorder level set
                self.no_rol = True
                self.current_rol = 0
        else:
            self.demand_dist = {}

        # Default distribution is not discrete
        self.discrete = True

        if self.dobr_error != 0 and printerror:
            print("DoBr: " + ERROR_CODES[self.dobr_error])

    def _val_oq(self):
        # Check the values of ioq and moq
        if self.dobr_error == 0 and validate_input(self.ioq, int_val=True):
            self.ioq = int(self.ioq)
        else:
            self.dobr_error = -9912
        if self.dobr_error == 0 and validate_input(self.moq, int_val=True):
            self.moq = max(int(round(self.moq)), self.ioq)
        else:
            self.dobr_error = -9914
        if self.dobr_error == 0 and (self.moq > self.ioq
              and abs(self.moq/self.ioq - int(self.moq/self.ioq)) > EPS):
            # Moq is not an integer multiple of ioq
            self.dobr_error = -9915

    def _val_shelflife(self):
        """"Validate the shelf life."""
        if self.dobr_error == 0:
            if self.shelflife != 0:
                self.dobr_error = -9999

    def _list_dists(self):
        # Determine which distribution cycles are needed
        # Basic cycles: L, R and R+L
        self.demand_dist = {"R": None, "L": None, "RL": None}
        # Additional cycles to determine the average ioh by integration
        if self.reviewperiod > 1:
            # For the trapezium rule: L+R-1
            self.demand_dist["RLm1"] = None
            if self.reviewperiod >= 3:
                # For Simpson's rule: L+(R-1)/2
                self.demand_dist["RLmid"] = None

    def _create_dists(self, usecp):
        """ Create the requested distributions in the dictionary from the
        base distribution.
        """
        # We will use convolution if L and R are (close to) integer
        # and we have a deterministic leadtime
        if (abs(self.leadtime-int(self.leadtime)) < EPS
                and abs(self.reviewperiod-int(self.reviewperiod)) < EPS
                and self.var_leadtime < EPS):
            # Use convolution
            self._convoluted_dist()
        else:
            # Generate required distributions with two-moment fits
            for cycle in self.demand_dist:
                cycle_periods = self._baseperiods(cycle)
                if cycle_periods == 0:
                    # Zero cycle length
                    self.demand_dist[cycle] = dobr_dist.ZeroDistArray()
                elif cycle_periods == 1 and self.var_leadtime < EPS:
                    # Re-use min period length
                    self.demand_dist[cycle] = self.base
                else:
                    self.demand_dist[cycle] = dobr_dist.select_dist(
                        cycle_periods*self.mean, self.vtm+self._vtmplus(cycle),
                        trycp=usecp)
                    if self.demand_dist[cycle] is None:
                        self.dobr_error = -9903

    def _convoluted_dist(self):
        # Check for cycles with a length equal to zero or the base cycle
        max_period = 0
        for cycle in self.demand_dist:
            cycle_periods = int(round(self._baseperiods(cycle)))
            if cycle_periods == 0:
                # Zero cycle length
                self.demand_dist[cycle] = dobr_dist.ZeroDistArray()
            elif cycle_periods == 1:
                # Base cycle length
                self.demand_dist[cycle] = self.base
            else:
                max_period = max(max_period, cycle_periods)
        # Remaining cycles with length >= 2
        if max_period >= 2:
            current_periods = 2
            tmp_pmf = self.base.apmf
            while current_periods <= max_period:
                tmp_pmf = convolve(tmp_pmf, self.base.apmf)
                for cycle in self.demand_dist:
                    cycle_periods = int(round(self._baseperiods(cycle)))
                    if cycle_periods == current_periods:
                        self.demand_dist[cycle] = dobr_dist.GeneratedDistArray(
                            cycle_periods*self.mean, self.vtm, tmp_pmf)
                current_periods += 1

    @staticmethod
    def adjust_rol(reorderlevel):
        """ For a discrete distribution, we require an integer reorder level. """
        return int(math.ceil(reorderlevel))

    def _adjust_shelfspace(self, capacity):
        # For a discrete distribution, we require an integer capacity
        return int(round(capacity))

    def _ordersize(self, reorderlevel, inv_pos):
        # Determine the ordersize based on the ip (inventory position).
        if inv_pos < reorderlevel:
            return (int((reorderlevel - 1 + self.moq - inv_pos)
                        / self.ioq) * self.ioq)
        return 0

    def mioh(self, reorderlevel):
        """ Return the maximum inventory on hand. """
        return reorderlevel - 1 + self._max_oq()

    def _create_dist_ip_plus(self):
        """"Create the distribution of the inventory position
        after potential ordering.
        """
        if self.moq <= self.ioq:
            # Uniform distributed (Hadley & Whitin, 1963)
            return self._create_dist_ip_plus_hw()
        if self.ioq == 1 and self.moq > 1:
            # Distribution based on Zheng & Federgruen (1991)
            return self._create_dist_ip_plus_zf()
        # Distribution based on Hill (EJOR 2006)
        return self._create_dist_ip_plus_hill()

    def _create_dist_ip_plus_hw(self):
        # Distribution of the inventory position after potential ordering
        # according to Hadley & Whitin (1963)
        dist_ip_plus = np.zeros(self.ioq)
        # Uniform distributed with probability 1/ioq
        dist_ip_plus.fill(1.0/self.ioq)
        return dist_ip_plus

    def _create_dist_ip_plus_zf(self):
        # Distribution of the inventory position after potential ordering
        # according to Zhang & Federgruen (1991)
        dist_ip_plus = np.zeros(self.moq)
        dist_r = self.demand_dist["R"]

        # RsS: order up to level = reorderLevel - 1 + MOQ
        # Start from S
        dist_ip_plus[self.moq-1] = 1./(1. - dist_r.pmf(0))
        sum_ip = dist_ip_plus[self.moq-1]
        x = self.moq - 1
        while x >= 1:
            dist_ip_plus[x-1] = 0.
            k = x + 1
            while k <= self.moq:
                dist_ip_plus[x-1] += dist_ip_plus[k-1]*dist_r.pmf(k-x)
                k += 1

            dist_ip_plus[x-1] = dist_ip_plus[x-1] * dist_ip_plus[self.moq-1]
            sum_ip += dist_ip_plus[x-1]
            x -= 1

        dist_ip_plus = dist_ip_plus / sum_ip
        return dist_ip_plus

    def _create_dist_ip_plus_zf_npopt(self):
        # Distribution of the inventory position after potential ordering
        # according to Zhang & Federgruen (1991)
        dist_ip_plus = np.zeros(self.moq)
        dist_r = self.demand_dist["R"]

        # RsS: order up to level = reorderLevel - 1 + MOQ
        # Start from S
        dist_ip_plus[self.moq-1] = 1./(1. - dist_r.pmf(0))
        sum_ip = dist_ip_plus[self.moq-1]
        x = self.moq - 1
        while x >= 1:
            u = max(0, self.moq-x-dist_r.ub())
            sum_x = np.sum(dist_ip_plus[x:self.moq-1-u]*dist_r.apmf[1:self.moq-x-u])

            dist_ip_plus[x-1] = sum_x * dist_ip_plus[self.moq-1]
            sum_ip += dist_ip_plus[x-1]
            x -= 1

        dist_ip_plus = dist_ip_plus / sum_ip
        return dist_ip_plus

    def _create_dist_ip_plus_hill(self):
        # Distribution of the inventory position after potential ordering
        # according to Hill (2006)

        # Relevant distribution
        dist_r = self.demand_dist["R"]

        # Fill transition matrix P
        p = np.zeros((self.moq, self.moq))
        for j in range(self.moq):
            for i in range(j, self.moq):
                if i-j <= dist_r.ub():
                    p[i, j] += dist_r.apmf[i-j]
        for i in range(self.moq):
            for j in range(self.moq-self.ioq, self.moq):
                k = 0
                while i+(self.moq-j)+k*self.ioq <= dist_r.ub():
                    p[i, j] += dist_r.apmf[i+(self.moq-j)+k*self.ioq]
                    k += 1

        # Matrix a and rhs-vector b for the set of linear equations Ax=b
        # Fill A with (I-P.T)
        mat_a = np.identity(self.moq) - p.T
        vec_b = np.zeros(self.moq)
        # Sum of probability
        mat_a[0, :] = 1.
        vec_b[0] = 1.

        try:
            # Solve the set of linear equations
            dist_ip_plus = np.linalg.solve(mat_a, vec_b)
            # Check the solution for negative values
            if any(x < 0. for x in dist_ip_plus):
                for i in range(self.moq):
                    dist_ip_plus[i] = max(0., dist_ip_plus[i])
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                self.dobr_error = -9981
                print("DoBr: " + ERROR_CODES[self.dobr_error])
                # Reset to uniform distribution of the IP
                dist_ip_plus = np.zeros(self.ioq)
                dist_ip_plus.fill(1.0/self.ioq)
            else:
                raise
        return dist_ip_plus

    def change_oq(self, ioq=1, moq=0):
        """ Change the order quantity parameters
        Keyword arguments:
        ioq          -- the incremental order quantity (default 1)
        moq          -- the minimal order quantity (default 0)
        """
        if self.dobr_error == 0:
            prev_ioq = self.ioq
            prev_moq = self.moq
            self.ioq = ioq
            self.moq = moq
            self._val_oq()
            if self.dobr_error == 0:
                if prev_ioq != self.ioq or prev_moq != self.moq:
                    # Parameters have changed
                    if self.dobr_error == 0:
                        # Create ip+ distribution (independent of reorder level)
                        self.inv_dist["IP+"] = dobr_dist.EmpiricalDistArray(
                            self._create_dist_ip_plus())
                        # Reset ioh distribution
                        self.no_rol = True
            else:
                print("DoBr: " + ERROR_CODES[self.dobr_error])
                # Reset
                self.dobr_error = 0
                self.ioq = prev_ioq
                self.moq = prev_moq


    def _create_dist_os(self):
        """Create the distribution of the order sizes."""
        mioh = self.mioh(self.current_rol)
        os_max = mioh
        dist_os = np.zeros(mioh+1)

        # Determine ip- from ip+ distribution
        dist_ip = self.inv_dist["IP+"]
        dist_d = self.demand_dist["R"]
        for ip in range(self.current_rol, mioh+1):
            for d in range(dist_d.ub()+1):
                os = int(self._ordersize(self.current_rol, ip-d)/self.ioq)
                dist_os[min(os, os_max)] += (dist_d.apmf[d]
                                             *dist_ip.apmf[ip-self.current_rol])
        return dist_os

    def fetch_dist(self, tau, reorderlevel):
        """"Return the requested distribution  as an array. """
        if tau == "OS":
            return self._create_dist_os()
        # Check if the requested distribution is already created
        dist_inv = self._create_inv_dist(reorderlevel, tau)
        if dist_inv is None:
            return dist_inv
        if tau == "IP+" and self.order_situation() == "back ordering":
            # Need to shift the IP
            mioh = self.mioh(self.current_rol)
            dist_ip = np.zeros(mioh+1)
            for i in range(self.current_rol, mioh+1):
                dist_ip[i] = self.inv_dist["IP+"].apmf[i - self.current_rol]
            return dist_ip
        # Return the pmf of the requested distribution
        return self.inv_dist[tau].apmf

    def _create_inv_dist(self, reorderlevel, tau):
        """Create the distribution of the inventory on-hand
        after (L) or before (RL) a potential delivery in case of backordering.
        """
        reorderlevel = self.adjust_rol(reorderlevel)
        if self.no_rol or reorderlevel != self.current_rol:
            # New or changed reorder level: reset IOH distributions
            self.inv_dist["IOH_L"] = None
            self.inv_dist["IOH_RL"] = None
            self.no_rol = False
            self.current_rol = reorderlevel

        if not (self.inv_dist[tau] is None):
            return self.inv_dist[tau]
        
        # Determine requested ioh distribution from ip+ distribution
        if tau == "IOH_L":
            dist_d = self.demand_dist["L"]
        else:
            dist_d = self.demand_dist["RL"]
        dist_ip = self.inv_dist["IP+"]
        mioh = self.mioh(reorderlevel)
        if mioh <= 0:
            # Never positive inventory
            dist_ioh = np.zeros(1)
            dist_ioh[0] = 1.
        else:
            # We temporary store the ioh in REVERSE order in tmp
            tmp = np.zeros(mioh+1)
            # Iterate over the part of the ip+ distribution that is
            # greater than zero
            for k in range(max(0,-reorderlevel), mioh-reorderlevel+1):
                # lb   : highest ioh that can be reached from this ip+ level
                lb = self._max_oq()-1-k
                # ub_i : lowest ioh that can be reached based on the size
                #        of the demand distribution
                ub_i = min(mioh, lb+dist_d.ub())
                # ub_d : range of the demand distibution used
                ub_d = min(mioh-lb, dist_d.ub())
                tmp[lb:ub_i+1] = np.add(tmp[lb:ub_i+1],
                                  np.multiply(dist_ip.apmf[k],
                                              dist_d.apmf[0:ub_d+1]))
            # The ioh is the reverse of the tmp array
            dist_ioh = np.flipud(tmp)
            for k in range(self._max_oq()):
                dist_ioh[0] += (dist_d.sf(reorderlevel + k + 1)
                                *dist_ip.apmf[k])
        # Check PMF for negative values
        if any(x < 0. for x in dist_ioh):
            for i in range(self._max_oq()):
                dist_ioh[i] = max(0., dist_ioh[i])
        # Store IOH distribution as DistArray
        self.inv_dist[tau] = dobr_dist.EmpiricalDistArray(dist_ioh)
        # Return the requested distribution
        return self.inv_dist[tau]

    def accuracy(self):
        """Generic accuracy of the KPIs in this class, based on shelflife
        and/or leadtime uncertainty."""
        # Generic accuracy of the KPIs in this class
        if self.shelflife > 0:
            return VAL_APPROX
        if self.leadtime > EPS and self.var_leadtime > EPS:
            return VAL_APPROX
        return VAL_EXACT

    def _supply_factor(self, reorderlevel, acc=False):
        # No loss with backordering
        if acc:
            return 1., VAL_EXACT
        return 1.

    def _prob_pos_demand(self, capacity):
        # Return probability positive demand above capacity
        return 1. - self.demand_dist["R"].cdf(capacity)

    def _eioh(self, cycle, reorderlevel, capacity=0, acc=False):
        # Determine the expected inventory on-hand
        eioh = 0.
        s = int(reorderlevel - capacity)
        dist_d = self.demand_dist[cycle]
        dist_ip = self.inv_dist["IP+"]
        if self._max_oq() == 1:
            # P[ip+=s] = 1
            eioh = dist_d.rloss(s)
        else:
            ub_d = dist_d.ub()
            # Determine lower bound iterator, due to range loss(i)
            # 0 <= s'+i and i>=0  -->  i = max(0, -s')
            lb_i = max(0, -s)
            if lb_i < self._max_oq():
                mu = dist_d.mean
                # First the sum product without the loss
                # SUM(i=0:q-1) (s-mu+i)*P[IP+=i]
                eioh = np.sum((s-mu+np.arange(lb_i, self._max_oq()))
                          *dist_ip.apmf[lb_i:self._max_oq()])
                # Next, add the sum product with the loss function
                # SUM(i=0:q-1) Loss(s+i)*P[IP+=i]
                if s+lb_i <= ub_d:
                    if s+(self._max_oq()-1) <= ub_d:
                        ub_i = self._max_oq()-1
                    else:
                        # Part of the ip range stays within the distribution range
                        # s+(q-1) > ub_d  --> ub_i = ub_d-s
                        ub_i = ub_d-s
                    eioh += np.sum(dist_d.aloss[s+lb_i:(s+ub_i)+1]
                               *dist_ip.apmf[lb_i:ub_i+1])
        # Return only positive values
        eioh = max(0., eioh)
        if acc:
            return eioh, self.accuracy()
        return eioh

    def _ebo(self, cycle, reorderlevel, eioh=None, acc=False):
        # Expected backorders based on the E[IOH] for the given cycle length
        # for Discrete distributed demand.
        if eioh is None:
            eioh = self._eioh(cycle, reorderlevel)
        ebo = max(0., eioh + self._mu_cycle(cycle) - self._eip(reorderlevel))
        if acc:
            return ebo, self.accuracy()
        return ebo

    def _eip(self, reorderlevel):
        # Expected inventory position after ordering
        dist_ip = self.inv_dist["IP+"]
        eip = 0
        for i in range(self._max_oq()):
            eip += (reorderlevel+i)*dist_ip.apmf[i]
        return eip

    def _poc(self, cycle, reorderlevel, capacity=0, acc=False):
        poc = 0.
        s = reorderlevel - capacity
        if self.mioh(s) > 0:
            dist_d = self.demand_dist[cycle]
            dist_ip = self.inv_dist["IP+"]
            if self._max_oq() == 1:
                # P[ip+=s] = 1
                poc = dist_d.cdf(s-1)
            else:
                ub_d = dist_d.ub()
                # Determine lower bound iterator, due to range cdf
                # 0 <= s'+i-1 and i>=0  -->  i = max(0, 1-s')
                # Note that lb_i < q-1
                lb_i = max(0, 1-s)
                # Does the ip range stay within the distribution range?
                if s+lb_i-1 <= ub_d:
                    if s+(self._max_oq()-1)-1 <= ub_d:
                        # Full ip range stays within the distribution range
                        ub_i = self._max_oq()-1
                    else:
                        # Part of the ip range stays within the distribution range
                        # s+(q-1)-1 > ub_d  --> ub_i = ub_d+1-s
                        ub_i = ub_d+1-s
                    # Take the sum product of the cdf with the ip probabilities
                    poc = np.sum(dist_d.acdf[s-1+lb_i:(s-1+ub_i)+1]
                                 *dist_ip.apmf[lb_i:ub_i+1])
                    if ub_i < self._max_oq()-1:
                        # All remaining ip values correspond to a cdf of 1
                        poc += np.sum(dist_ip.apmf[ub_i+1:])
                else:
                    # Ip range past the upper bound of the distribution
                    poc += np.sum(dist_ip.apmf[lb_i:])
        # Return only positive values
        poc = max(0., poc)
        if acc:
            return poc, self.accuracy()
        return poc

    def _encr(self, reorderlevel, capacity, acc=False):
        # Return the expected number of concurrent backroom trips
        dist_d = self.demand_dist["R"]
        mioh = self.mioh(reorderlevel)
        dist_i = self._create_inv_dist(reorderlevel, "IOH_L")

        encr = 0.
        ioh_l = capacity + 1
        while ioh_l <= mioh:
            n = 0
            while n*capacity < ioh_l:
                n += 1
                # Demand depletes capacity
                encr += n*(dist_d.sf(n*capacity+1)
                           -dist_d.sf((n+1)*capacity+1))*dist_i.apmf[ioh_l]
            ioh_l += 1
        # Return only positive values
        encr = max(0., encr)
        if acc:
            return encr, self.accuracy()
        return encr

    def _eol(self, reorderlevel, acc=False):
        # Return the expected number of order lines per review period
        # for discrete demand.
        eol = 0.
        # Relevant distributions
        dist_d = self.demand_dist["R"]
        dist_ip = self.inv_dist["IP+"]
        ub_i = min(self._max_oq(), dist_d.ub()+1)
        for i in range(ub_i):
            eol += dist_d.sf(i+1)*dist_ip.apmf[i]
        # Return only positive values
        eol = max(0., eol)
        if acc:
            return eol, VAL_EXACT
        return eol

    def _ez(self, reorderlevel, acc=False):
        # Expected relative outdating z not known yet for discrete demand
        # with backordering
        ez = 0.
        if acc:
            return ez, VAL_NOTYET
        return ez

    def _eua(self, reorderlevel, unitcap, acc=False):
        # Return the expected number of unit arrivals per review period
        # for discrete demand.
        eua = 0.
        # Capacity is the maximum of the IOQ and the unit load
        c = max(self.ioq, unitcap)
        # Relevant distributions
        dist_d = self.demand_dist["R"]
        dist_ip = self.inv_dist["IP+"]

        # First: summation over the IP+ distribution
        ub_i = min(self._max_oq(), dist_d.ub()+1)
        for i in range(ub_i):
            inc_eua = 0.
            # Initial number of ioq's
            n = int(self.moq/self.ioq)
            # Second: summation over the demand during the review period
            d_r = i + 1
            while d_r <= dist_d.ub():
                inc_eua += (math.ceil(n*self.ioq/c)
                            *(dist_d.cdf(d_r+self.ioq-1)-dist_d.cdf(d_r-1)))
                # Increase with step size ioq
                d_r += self.ioq
                n += 1
            # Multiply with the IP+ probability
            eua += inc_eua*dist_ip.apmf[i]
        # Return only positive values
        eua = max(0., eua)
        if acc and self.ioq < 1.:
            return eua, VAL_APPROX
        if acc:
            return eua, VAL_EXACT
        return eua

    def _enb(self, reorderlevel, acc=False):
        # Expected number of batches
        # for Discrete distributed demand.

        mioh = self.mioh(reorderlevel)
        # Relevant distributions
        dist_i = self._create_inv_dist(reorderlevel, "IOH_RL")

        eol = self._eol(reorderlevel)
        if eol > EPS:
            eos = self.mean*self.reviewperiod/eol
        else:
            eos = self._max_oq()

        enb = 0.
        for i in range(mioh+1):
            enb += math.ceil(i/eos)*dist_i.apmf[i]
        # Add the expected order lines eol
        enb += eol
        # Return only positive values
        enb = max(0., enb)
        if acc:
            return enb, VAL_APPROX
        return enb

    def _eusl(self, reorderlevel, unitcap, eos, acc=False):
        # Expected number of unit load storage locations
        # for Discrete distributed demand.
        eusl = 0.
        # Relevant distributions
        dist_i = self._create_inv_dist(reorderlevel, "IOH_RL")

        # Unit capacity is the maximum of the IOQ and the unit load
        c = max(self.ioq, unitcap)
        # Batch size is the minimum of unitcap and expected order size
        c = min(eos, c)

        mioh = self.mioh(reorderlevel)
        for i in range(mioh+1):
            eusl += math.ceil(i/c)*dist_i.apmf[i]
        # Return only positive values
        eusl = max(0., eusl)
        if acc:
            return eusl, VAL_APPROX
        return eusl

    def dist_name(self):
        """Return the fitted distribution for the review period."""
        return self.demand_dist["R"].dist_name()


class InvSysDiscreteLS(InvSysDiscreteBO):
    """ A class used to describe (R,s,nQ,S) inventory systems.
    The methods of this subclass assume (mixtures of) discrete demand
    distributions, lost sales, and only the reorder level as parameter.
    """

    def __init__(
            self, mean_perioddemand, stdev_perioddemand, leadtime,
            stdev_leadtime=0., reviewperiod=1, ioq=1, moq=0, shelflife=0,
            printerror=True, empiricalpmf=None, usecp=False):
        """ Initialize

        Keyword arguments:
        stdev_leadtime  -- the standard deviation of the leadtime (default 0.0)
        reviewperiod    -- the time between two reviews (default 1.0)
        ioq             -- the incremental order quantity (default 1)
        moq             -- the minimal order quantity (default 0)
        shelflife       -- the remaining shelflife upon entering the stock point
        printerror      -- print error messages to the console (default True)
        empiricalpmf    -- the empirical pmf for the base period (default None)
        usecp           -- use Compound Poisson if VTM>1 (default False)
        """

        super().__init__(
            mean_perioddemand, stdev_perioddemand, leadtime,
            stdev_leadtime=stdev_leadtime, reviewperiod=reviewperiod,
            ioq=ioq, moq=moq, shelflife=shelflife,
            printerror=printerror, empiricalpmf=empiricalpmf, usecp=usecp)

        # We can not deal yet with L>R
        # TEMP: allow LS2
        # if self.leadtime > self.reviewperiod and self._max_oq() > 1:
        #     self.dobr_error = -9999

        if self.dobr_error == 0:
            self.lost_sales = True
            if self.shelflife > 0:
                # Load regression coefficients
                try:
                    with open("zreg_fifo.json", encoding="utf8") as fp_zreg:
                        self.zreg_fifo = json.load(fp_zreg)
                except OSError:
                    # File with regression coeffients not found
                    self.dobr_error = -9980
                    if printerror:
                        print("DoBr: " + ERROR_CODES[self.dobr_error])
                    self.zreg_fifo = None
            else:
                self.zreg_fifo = None

            # No specific ip distribution yet: re-use backorder ip+ dist
            self.dist_ip = None
        else:
            self.demand_dist = {}
            if printerror:
                print("DoBr: " + ERROR_CODES[self.dobr_error])

    def _list_dists(self):
        # Determine which demand distribution cycles are needed
        # Basic cycles: L, R and R+L
        self.demand_dist = {"R": None, "L": None, "RL": None}
        # Additional cycles to determine the average ioh by integration
        if self.reviewperiod > 1:
            # For the trapezium rule; L+R-1
            self.demand_dist["Rm1"] = None
            if self.reviewperiod >= 3:
                # For Simpson's rule: L+(R-1)/2
                self.demand_dist["Rmid"] = None
        # Specific cycles for lost sales
        self.demand_dist["Rfrac"] = None       # Fraction R
        self.demand_dist["L1R"] = None         # Bijvank & Johansen
        # Specific cycles for perishables
        if self.shelflife > 0:
            self.demand_dist["Rho"] = None     # EZ
            self.demand_dist["LM"] = None      # EZ_B
            self.demand_dist["Mm0"] = None     # EZ_FCC
            if self.shelflife > 1:
                self.demand_dist["Mm1"] = None
                if self.shelflife > 2:
                    self.demand_dist["Mm2"] = None
                    if self.shelflife > 3:
                        self.demand_dist["Mm3"] = None

    def _val_shelflife(self):
        """"Validate the shelf life."""
        if self.dobr_error == 0:
            if self.shelflife > 0 and self.shelflife <= self.reviewperiod:
                self.dobr_error = -9930
            elif abs(self.shelflife - int(self.shelflife)) > EPS:
                self.dobr_error = -9931
            else:
                self.shelflife = min(30, self.shelflife)

    @staticmethod
    def order_situation():
        return "lost sales"

    def calc_kpis(self, kpi_list, reorderlevel,
                  unitcap=0, capacity=0):
        """ Return the value of the KPIs in the passed list.

        Keyword arguments:
        unitcap  -- the unit load capacity (default = 0.)
        capacity -- the capacity (default 0)
        """
        if self.leadtime <= self.reviewperiod:
            return super().calc_kpis(kpi_list, reorderlevel, 
                              unitcap=unitcap, capacity=capacity)
        else:
            kpi_results = {}
            for kpi in kpi_list:
                # Reset to NOT APPLIC
                kpi_results[kpi] = [0., VAL_NOTAPPLIC]
                # Only calculate if requested
                if kpi == "Fillrate":
                    kpi_results[kpi] = self.fillrate(reorderlevel, acc=True)
                elif kpi == "Readyrate":
                    kpi_results[kpi] = self.readyrate(reorderlevel, acc=True)
                elif kpi == "ELT" and self.var_leadtime > EPS:
                    kpi_results[kpi] = [self.leadtime, VAL_EXACT]
                elif kpi == "EIOH_L":
                    # kpi_results[kpi] = [0., VAL_NOTYET]
                    kpi_results[kpi] = self.eioh_l(reorderlevel, acc=True)
                elif kpi == "EIOH_RL":
                    # kpi_results[kpi] = [0., VAL_NOTYET]
                    kpi_results[kpi] = self.eioh_rl(reorderlevel, acc=True)
                elif kpi == "EOL":
                    kpi_results[kpi] = self.eol(reorderlevel=reorderlevel, acc=True)
                elif kpi == "EOS":
                    kpi_results[kpi] = self.eos(reorderlevel=reorderlevel, acc=True)
                elif kpi == "ESUP":
                    kpi_results[kpi] = self.supply(reorderlevel=reorderlevel, acc=True)
                elif kpi == "EUA":
                    kpi_results[kpi] = [0., VAL_NOTYET]
                    # kpi_results[kpi] = self.eua(reorderlevel=reorderlevel,
                    #                           unitcap=unitcap, acc=True)
                elif kpi == "ENB":
                    kpi_results[kpi] = [0., VAL_NOTYET]
                    # kpi_results[kpi] = self.enb(reorderlevel=reorderlevel, acc=True)
                elif kpi == "EUSL_L":
                    kpi_results[kpi] = [0., VAL_NOTYET]
                    # kpi_results[kpi] = self.eusl_l(reorderlevel=reorderlevel,
                    #                           unitcap=unitcap, acc=True)
                elif kpi == "EW" and self.shelflife > 0:
                    kpi_results[kpi] = self.ew(reorderlevel, acc=True)
                elif kpi == "EST":
                    kpi_results[kpi] = [0., VAL_NOTYET]
                    # kpi_results[kpi] = self.est(reorderlevel, acc=True)
                elif kpi == "POC" and capacity > 0:
                    kpi_results[kpi] = [0., VAL_NOTYET]
                    # kpi_results[kpi] = self.poc(reorderlevel,
                    #                          capacity=capacity, acc=True)
                elif kpi == "EIBR" and capacity > 0:
                    kpi_results[kpi] = [0., VAL_NOTYET]
                    # kpi_results[kpi] = self.eibr(reorderlevel,
                    #                           capacity=capacity, acc=True)
                elif kpi == "ENIR" and capacity > 0:
                    kpi_results[kpi] = [0., VAL_NOTYET]
                    # kpi_results[kpi] = self.enir(reorderlevel,
                    #                           capacity=capacity, acc=True)
                elif kpi == "ENCR" and capacity > 0:
                    kpi_results[kpi] = [0., VAL_NOTYET]
                    # kpi_results[kpi] = self.encr(reorderlevel,
                    #                          capacity=capacity, acc=True)
            return kpi_results

    def _create_inv_dist(self, reorderlevel, tau):
        # Has the reorder level changed?
        reorderlevel = self.adjust_rol(reorderlevel)
        if self.no_rol or reorderlevel != self.current_rol:
            # New or changed reorder level: reset IOH and IP distributions
            self.inv_dist["IOH_L"] = None
            self.inv_dist["IOH_RL"] = None
            self.inv_dist["IP-"] = None
            self.inv_dist["IP+"] = None
            self.no_rol = False
            self.current_rol = reorderlevel

        # noo = self.leadtime*self.mean/max(self.reviewperiod*self.mean,
        #                                   self._max_oq())
        if self.leadtime < EPS:
            # First, check if the ioh_l distribution exists
            if self.inv_dist["IOH_L"] is None:
                # Store IOH distribution as DistArray
                self.inv_dist["IOH_L"] = dobr_dist.EmpiricalDistArray(
                    self._create_inv_dist_zerolt(reorderlevel))
                # Since ioh_l = IP+, we can set this too
                self.inv_dist["IP+"] = self.inv_dist["IOH_L"]
            if tau == "IP-" and self.inv_dist["IP-"] is None:
                self.inv_dist["IP-"] = dobr_dist.EmpiricalDistArray(
                    self._create_inv_dist_ip_min(reorderlevel))
        elif self.leadtime <= self.reviewperiod:
            # First, check if the ioh_l distribution exists
            if self.inv_dist["IOH_L"] is None:
                # Store IOH distribution as DistArray
                self.inv_dist["IOH_L"] = dobr_dist.EmpiricalDistArray(
                    self._create_inv_dist_fraclt(reorderlevel))
            if self.leadtime == self.reviewperiod:
                # Since ioh_l = ip-, we can set this too
                self.inv_dist["IP-"] = self.inv_dist["IOH_L"]
            else:
                if (tau in ("IP-", "IP+")
                      and self.inv_dist["IP-"]) is None:
                    self.inv_dist["IP-"] = dobr_dist.EmpiricalDistArray(
                        self._create_inv_dist_ip_min(reorderlevel,
                                                     tau="IOH_L"))
            if tau == "IP+" and self.inv_dist["IP+"] is None:
                # Determine the ip+ from the ip-
                self.inv_dist["IP+"] = dobr_dist.EmpiricalDistArray(
                    self._create_inv_dist_ip_plus(reorderlevel))
        elif self._max_oq() == 1:
            if self.inv_dist["IOH_L"] is None:
                # Store IOH distribution as DistArray
                self.inv_dist["IOH_L"] = dobr_dist.EmpiricalDistArray(
                    self._create_inv_dist_bj(reorderlevel))
                # TEMP: IP+ is fixed at s
                mioh = self.mioh(reorderlevel)
                dist_ip = np.zeros(mioh+1)
                dist_ip[mioh] = 1.
                self.inv_dist["IP+"] = dobr_dist.EmpiricalDistArray(dist_ip)
            if tau == "IP-" and self.inv_dist["IP-"] is None:
                self.inv_dist["IP-"] = dobr_dist.EmpiricalDistArray(
                    self._create_inv_dist_ip_min(reorderlevel))

        if self.inv_dist["IOH_L"] is None:
            # No distribution available yet
            return None
        # Second, create dependent distribution ioh_rl using ioh_l
        if tau == "IOH_RL" and self.inv_dist["IOH_RL"] is None:
            self.inv_dist["IOH_RL"] = dobr_dist.EmpiricalDistArray(
                self._create_inv_dist_rl(reorderlevel))
        return self.inv_dist[tau]

    def _create_inv_dist_rl(self, reorderlevel):
        # Create the distribution of the inventory on-hand
        # before a potential delivery from the ioh_l.
        dist_d = self.demand_dist["R"]
        dist_i = self.inv_dist["IOH_L"]
        mioh = self.mioh(reorderlevel)
        # We temporary store the ioh in REVERSE order in tmp
        tmp = np.zeros(mioh+1)
        # Iterate over the part of the ip+ distribution that is
        # greater than zero
        for k in range(mioh+1):
            # lb   : highest ioh that can be reached from this level
            lb = mioh - k
            # ub_i : lowest ioh that can be reached based on the size
            #        of the demand distribution
            ub_i = min(mioh, lb+dist_d.ub())
            # ub_d : range of the demand distibution used
            ub_d = min(mioh-lb, dist_d.ub())
            tmp[lb:ub_i+1] = np.add(tmp[lb:ub_i+1],
                              np.multiply(dist_i.apmf[k],
                                          dist_d.apmf[0:ub_d+1]))
        # The ioh is the reverse of the tmp array
        dist_ioh = np.flipud(tmp)
        for i in range(mioh+1):
            dist_ioh[0] += (dist_d.sf(i+1)*dist_i.apmf[i])
        return dist_ioh

    def _create_inv_dist_ip_min(self, reorderlevel, tau="IP+"):
        # Create the distribution of the inventory position
        # before ordering from the ip+.
        if tau == "IP+":
            dist_d = self.demand_dist["R"]
            dist_i = self.inv_dist["IP+"]
        elif tau == "IOH_L":
            dist_d = self.demand_dist["Rfrac"]
            dist_i = self.inv_dist["IOH_L"]

        mioh = self.mioh(reorderlevel)
        # We temporary store the ip in REVERSE order in tmp
        tmp = np.zeros(mioh+1)
        # Iterate over the part of the ip+ distribution that is
        # greater than zero
        for k in range(mioh+1):
            # lb   : highest ip that can be reached from this level
            lb = mioh - k
            # ub_i : lowest ip that can be reached based on the size
            #        of the demand distribution
            ub_i = min(mioh, lb+dist_d.ub())
            # ub_d : range of the demand distibution used
            ub_d = min(mioh-lb, dist_d.ub())
            tmp[lb:ub_i+1] = np.add(tmp[lb:ub_i+1],
                              np.multiply(dist_i.apmf[k],
                                          dist_d.apmf[0:ub_d+1]))
        # The ip is the reverse of the tmp array
        dist_ip = np.flipud(tmp)
        for i in range(mioh+1):
            dist_ip[0] += (dist_d.sf(i+1)*dist_i.apmf[i])
        return dist_ip

    def _create_inv_dist_ip_plus(self, reorderlevel):
        # Create the distribution of the inventory position
        # after ordering from the ip-.
        dist_i = self.inv_dist["IP-"]
        mioh = self.mioh(reorderlevel)
        dist_ip = np.zeros(mioh+1)
        for ip in range(mioh+1):
            os = self._ordersize(reorderlevel, ip)
            dist_ip[ip+os] += dist_i.apmf[ip]

        return dist_ip

    def _create_inv_dist_zerolt(self, reorderlevel):
        # Determine for L=0 the IohL (= IP+)

        # Determine relevant distributions
        dist_r = self.demand_dist["R"]
        mioh = self.mioh(reorderlevel)
        dist_ioh = np.zeros(mioh+1)

        if self._max_oq() > 1:
            # Fill transition matrix P (IP+ = IohL --> IP+)
            p = np.zeros((self._max_oq(), self._max_oq()))
            for i in range(self._max_oq()):
                ub_i = min(reorderlevel+i, dist_r.ub())
                for x in range(ub_i+1):
                    os = self._ordersize(reorderlevel, reorderlevel+i-x)
                    j = i + os - x
                    if x == reorderlevel+i:
                        # Demand depletes ioh
                        p[i, j] += dist_r.sf(x)
                    else:
                        p[i, j] += dist_r.pmf(x)

            # Matrix a and rhs-vector b for the set of linear equations Ax=b
            # Fill A with (I-P.T)
            mat_a = np.identity(self._max_oq()) - p.T
            vec_b = np.zeros(self._max_oq())
            # Sum of probability
            mat_a[0, :] = 1.
            vec_b[0] = 1.

            try:
                # Solve the set of linear equations
                range_ioh = np.linalg.solve(mat_a, vec_b)
                range_ioh = dobr_dist.check_pmf(range_ioh)
                # Fill the ioh_l
                for k in range(self._max_oq()):
                    dist_ioh[reorderlevel+k] = range_ioh[k]
            except np.linalg.LinAlgError as err:
                if "Singular matrix" in str(err):
                    self.dobr_error = -9982
                    print("DoBr: " + ERROR_CODES[self.dobr_error])
                    input("Press key...")
                    # Reset to (R,S)
                    dist_ioh[mioh] = 1.
                else:
                    raise
        else:
            # IP+ is one
            dist_ioh[mioh] = 1.
        return dist_ioh

    def _create_inv_dist_fraclt(self, reorderlevel):
        # Determine for L<=R the IohL
        # Maximum inventory on hand
        mioh = self.mioh(reorderlevel)
        dist_ioh = np.zeros(mioh+1)

        # P_L: Transitions from ordering moment to delivery after the lead-time
        dist_l = self.demand_dist["L"]
        p_l = np.zeros((mioh+1, mioh+1))
        for i in range(mioh+1):
            os = self._ordersize(reorderlevel, i)
            for j in range(mioh+1):
                if j < os or j > i+os:
                    # Unreachable
                    p_l[i, j] = 0.
                elif j == os:
                    # IOH gets zero
                    p_l[i, j] = dist_l.sf(i)
                else:
                    # No OOS during lead-time
                    p_l[i, j] = dist_l.pmf(i+os-j)
        # Remaining part of review period?
        r_frac = self._baseperiods("Rfrac")
        if r_frac > 0:
            dist_rf = self.demand_dist["Rfrac"]
            # p_rf: Transitions from delivery moment
            # to delivery after the lead-time
            p_rf = np.zeros((mioh+1, mioh+1))
            for i in range(mioh+1):
                for j in range(mioh+1):
                    if i < j:
                        # Unreachable (only downward!)
                        p_rf[i, j] = 0.
                    elif j == 0:
                        # IOH gets zero during R-L
                        p_rf[i, j] = dist_rf.sf(i)
                    else:
                        p_rf[i, j] = dist_rf.pmf(i-j)

            # The transition from ioh_l to ioh_l proceeds in two steps:
            # first p_rf (to reach the inventory position before ordering)
            # and next p_l (to reach the ioh_l): p=p_rf*p_l
            p = np.matmul(p_rf, p_l)
        else:
            p = p_l

        # Matrix a and rhs-vector b for the set of linear equations Ax=b
        # fill A with (I-P.T)
        a = np.identity(mioh+1) - p.T
        b = np.zeros(mioh+1)
        # Sum of probability
        a[mioh, :] = 1.
        b[mioh] = 1.

        try:
            # Solve the set of linear equations
            res_ioh = np.linalg.solve(a, b)
            res_ioh = dobr_dist.check_pmf(res_ioh)
            for ioh_l in range(mioh+1):
                dist_ioh[ioh_l] = res_ioh[ioh_l]
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                self.dobr_error = -9982
                print("DoBr: " + ERROR_CODES[self.dobr_error])
                # Reset to (R,S)
                dist_ioh[mioh] = 1.
            else:
                raise

        return dist_ioh

    def _create_inv_dist_bj(self, reorderlevel):
        # Determine relevant distributions
        dist_l = self.demand_dist["L"]
        # Maximum inventory on hand
        mioh = self.mioh(reorderlevel)
        dist_ioh = np.zeros(mioh+1)

        # Determine cs factor
        denom = ((int(self.leadtime/self.reviewperiod) + 1)
                 *(super()._eioh("L", reorderlevel)
                   - super()._eioh("RL", reorderlevel))
                 + super()._eioh("L1R", reorderlevel))
        cs = reorderlevel / denom
        eioh_l = 0
        for i in range(1, reorderlevel+1):
            dist_ioh[i] = cs*dist_l.pmf(reorderlevel-i)
            eioh_l += i*dist_ioh[i]
        dist_ioh[0] += cs*dist_l.sf(reorderlevel+1)
        # Correct for negative ioh values?
        dist_ioh[0] = max(0., dist_ioh[0])
        excess_pmf = np.sum(dist_ioh) - 1.0
        if excess_pmf > EPS:
            redux = min(excess_pmf, dist_ioh[0])
            dist_ioh[0] -= redux
            excess_pmf -= redux
        x = 1
        while excess_pmf > EPS and x < mioh:
            n = mioh - x + 1
            shift_factor = x/(n*x+n*(n+1)/2)
            redux = min(excess_pmf*(1 + n*shift_factor), dist_ioh[x])
            shift = redux*shift_factor
            if redux > EPS:
                dist_ioh[x] -= redux
                add_pmf = 0.
                for i in range(x, mioh+1):
                    dist_ioh[i] += shift
                    add_pmf += shift
            eioh_l = 0
            for i in range(1, reorderlevel+1):
                eioh_l += i*dist_ioh[i]
        
            excess_pmf = np.sum(dist_ioh) - 1.0
            x += 1
        # Remove remainder pmf
        x = 1
        while excess_pmf > EPS:
            redux = min(excess_pmf, dist_ioh[x])
            dist_ioh[x] -= redux
            excess_pmf -= redux
            x += 1

        return dist_ioh

    def _create_inv_dist_arts(self, reorderlevel):
        # Determine relevant distributions
        #dist_l = self.demand_dist["L"]
        dist_r = self.demand_dist["R"]
        s = reorderlevel
        dist_ioh_l = np.zeros(s+1)

        p = np.zeros((s+1, s+1))
        for i in range(s+1):
            # j < S
            for j in range(s):
                for k in range(j+1):
                    p[i, j] += dist_r.pmf(k)*dist_r.pmf(i+k-j)
            # j = S
            for k in range(i+1):
                p[i, s] += dist_r.pmf(k)*dist_r.sf(s+k-i)

        # Matrix a and rhs-vector b for the set of linear equations Ax=b
        # fill A with (I-P.T)
        a = np.identity(s+1) - p.T
        b = np.zeros(s+1)
        # Sum of probability
        a[s, :] = 1.
        b[s] = 1.

        try:
            # Solve the set of linear equations
            dist_pipe = np.linalg.solve(a, b)
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                self.dobr_error = -9982
                print("DoBr: " + ERROR_CODES[self.dobr_error])
                # Reset
                dist_ioh_l[s] = 1.
            else:
                raise
        ioh_l = 0
        while ioh_l <= s:
            dist_ioh_l[ioh_l] = dist_pipe[s-ioh_l]
            ioh_l += 1

        return dist_ioh_l

    def _create_dist_os(self):
        """Create the distribution of the order sizes."""
        os_max = self.mioh(self.current_rol)
        dist_os = np.zeros(os_max+1)

        dist_ip_min = self._create_inv_dist(self.current_rol, "IP-")
        if dist_ip_min is None:
            return None
        for ip in range(self.current_rol):
            os = int(self._ordersize(self.current_rol, ip)/self.ioq)
            dist_os[min(os, os_max)] += dist_ip_min.apmf[ip]
        return dist_os

    def accuracy(self):
        """Generic accuracy of the KPIs in this class, based on shelflife
        leadtime uncertainty and/or number of outstanding orders."""
        if self.shelflife > 0:
            return VAL_APPROX
        if self.leadtime > EPS and self.var_leadtime > EPS:
            return VAL_APPROX
        if self.leadtime > self.reviewperiod:
            # More than 1 outstanding order
            return VAL_APPROX
        return VAL_EXACT

    def _eioh(self, cycle, reorderlevel, capacity=0, acc=False):
        eioh = 0.
        if reorderlevel > 0:
            if self.leadtime <= self.reviewperiod:
                # Create ioh_l distribution
                dist_i = self._create_inv_dist(reorderlevel, "IOH_L")
                for ioh in range(capacity+1, self.mioh(reorderlevel)+1):
                    if cycle == "L":
                        revloss = ioh
                    elif cycle == "RL":
                        revloss = self.demand_dist["R"].rloss(ioh)
                    elif cycle == "RLm1":
                        revloss = self.demand_dist["Rm1"].rloss(ioh)
                    elif cycle == "RLmid":
                        revloss = self.demand_dist["Rmid"].rloss(ioh)
                    elif cycle == "LM":
                        revloss = self.demand_dist["LM"].rloss(ioh)
                    eioh += (revloss - capacity)*dist_i.apmf[ioh]
            else:
                fr = self._fr(reorderlevel)
                eioh_l_bo = super()._eioh("L", reorderlevel)
                eioh_rl_bo = super()._eioh("RL", reorderlevel)
                eioh_rl = (eioh_rl_bo*fr*self.reviewperiod*self.mean
                           /(eioh_l_bo - eioh_rl_bo))
                if cycle =="RL":
                    eioh = eioh_rl
                elif cycle == "L":
                    eioh = fr*self.reviewperiod*self.mean + eioh_rl
                    
            # Return only positive values
            eioh = max(0., eioh)
            if acc:
                return eioh, self.accuracy()
            return eioh
        # With lost sales, E[IOH] is zero for reorder levels less or equal to zero
        if acc:
            return eioh, VAL_EXACT
        return eioh

    def _ebo(self, cycle, reorderlevel, eioh=None, acc=False):
        # No backorders
        if acc:
            return 0., VAL_EXACT
        return 0.

    def _poc(self, cycle, reorderlevel, capacity=0, acc=False):
        # We only have overflow if the maximum inventory on hand (MIOH)
        # is greater than the capacity
        poc = 0.
        mioh = self.mioh(reorderlevel)
        if self.leadtime > self.reviewperiod:
            # Not yet implemented
            if acc:
                return poc, VAL_NOTYET
            return poc
        if reorderlevel > 0 and mioh > capacity:
            if cycle == "L":
                dist_i = self._create_inv_dist(reorderlevel, "IOH_L")
            elif cycle == "RL":
                dist_i = self._create_inv_dist(reorderlevel, "IOH_RL")
            poc = np.sum(dist_i.apmf[capacity+1:mioh+1])
            # Return only positive values
            poc = max(0., poc)
            if acc:
                return poc, self.accuracy()
            return poc
        if acc:
            return poc, VAL_EXACT
        return poc

    def _eol(self, reorderlevel, acc=False):
        # Return the expected number of order lines per review period
        # for discrete demand.
        eol = 0.
        if reorderlevel <= 0:
            if acc:
                return eol, VAL_EXACT
            return eol
        # No approximations yet for L>R
        if self.leadtime > self.reviewperiod:
            mu_r = self._supply_factor(reorderlevel)*self.mean*self.reviewperiod
            eol = mu_r/max(self._max_oq(), mu_r)
            if acc:
                return eol, VAL_APPROX
            return eol
        # Other cases
        if (self.shelflife > 0
            and self._max_oq()/(self.mean*self.shelflife) >= 1.):
            # Special case for perishables with FCC>=1
            # Large order size
            os_max = math.ceil(reorderlevel/self._max_oq())*self._max_oq()
            u = min(FCC_UMAX, self.shelflife-1)
            p_cum = 0.
            eol = 0.
            while u > 0:
                p_order = (1.0
                    - self.demand_dist["Mm" + str(u)].cdf(os_max-reorderlevel)
                    - p_cum)
                eol += p_order/(self.shelflife-u+1)
                p_cum += p_order
                u -= 1

            if reorderlevel <= self.leadtime*self.mean:
                # No EWA: we can order at t=m
                eol += (1.0 - p_cum)/(self.shelflife+self.leadtime)
            else:
                # We never order at t=m
                eol += (1.0 - p_cum)/self.shelflife
        else:
            # Check if ip- distribution is available
            dist_ip_min = self._create_inv_dist(reorderlevel, "IP-")
            eol = dist_ip_min.cdf(reorderlevel-1)
        # Return only positive values
        eol = max(0., eol)
        if acc:
            return eol, self.accuracy()
        return eol

    def _supply_factor(self, reorderlevel, acc=False):
        if reorderlevel > 0:
            sf = self._fr(reorderlevel) + self._ez(reorderlevel)
            if acc:
                return sf, self.accuracy()
            return sf
        if acc:
            return 0., VAL_EXACT
        return 0.

    def _ez_a(self, reorderlevel):
        # Determine zA
        rho = int(self.shelflife/self.reviewperiod)*self.reviewperiod
        bound = math.ceil((
            reorderlevel
            - round(self.mean * (self.leadtime
                                 + self.shelflife - rho) + EPS, 0))
                          / self._max_oq()) * self._max_oq()
        bound = max(bound, self._max_oq())
        z_a = max(0., self.demand_dist["Rho"].rloss(bound))
        return z_a/(rho*self.mean)

    def _ez_b(self, reorderlevel):
        # Determine zB
        rho = int(self.shelflife/self.reviewperiod)*self.reviewperiod
        z_b = 0.
        for q in range(self._max_oq()):
            ip = reorderlevel + q
            z_b += self.demand_dist["LM"].rloss(ip)/self.ioq
        return z_b/(rho*self.mean)

    def _fr_bj(self, reorderlevel):
        """ Fill rate approximation based on Bijvank & Johansen (EJOR, 2012). """
        eioh_l_bo = super()._eioh("L", reorderlevel)
        eioh_rl_bo = super()._eioh("RL", reorderlevel)
        # Determine cs factor
        denom = ((int(self.leadtime/self.reviewperiod) + 1)
                 *(eioh_l_bo - eioh_rl_bo)
                 + super()._eioh("L1R", reorderlevel))
        # Cs factor is modified to deal with Q>1
        cs = (reorderlevel + (self._max_oq()-1)/2) / denom
        
        return cs*(eioh_l_bo - eioh_rl_bo)/(self.reviewperiod*self.mean)

    def _fr_ls3(self, reorderlevel):
        """ Fill rate approximation LS3, based on 
        Van Donselaar & Broekmeulen (IJPE, 2013). 
        """
        # First, determine the lower bound based on backordering
        test = InvSysDiscreteBO(self.mean, self.variance**0.5,
                                self.leadtime, ioq=self.ioq)
        fr_lb = max(EPS*self.mean, test.fillrate(reorderlevel))
        # Two regimes, based on number of outstanding orders (noo) and CoV
        noo = self.leadtime*self.mean/max(self._max_oq(), self.mean*self.reviewperiod)
        if noo >= 5:
            # Regression modification of backorder fill rate
            c_rl = math.sqrt(self.variance/(self.leadtime + self.reviewperiod))/self.mean
            c_rl_factor = 1.3218*(c_rl**(-0.552))
            return (fr_lb - 1.0172 - c_rl_factor)/c_rl_factor
        # Fill rate approximation based on noo
        EPS_LS2 = 0.005
        nr_iterations = 0
        fr = fr_lb
        fr_prev = 1.
        delta = 1.
        # delta_prev = 1000.
        while True:
            nr_iterations += 1
            fr_guess = fr
            fr_prev = fr
            # delta_prev = delta
            test = InvSysDiscreteBO(fr_guess*self.mean,
                                    (self.vtm*fr_guess*self.mean)**0.5,
                                    self.leadtime, ioq=self.ioq)
            fr = test.fillrate(reorderlevel)
            if fr < EPS:
                fr = fr_prev
            delta = abs(fr - fr_prev)
            if delta < EPS_LS2 or nr_iterations >= 50:
                break

        if nr_iterations >= 50:
            fr = (fr + fr_prev)/2

        # Regression modification
        return min(1., (fr + 0.062 * noo - 0.128) / (0.062 * noo + 0.87))

    def _fr(self, reorderlevel, eioh_l=None, eioh_rl=None):
        if self.leadtime > self.reviewperiod:
            ss = reorderlevel - (self.leadtime+self.reviewperiod)*self.mean
            if self._max_oq() == 1 or ss < 0.:
                fr = self._fr_bj(reorderlevel)
            else:
                fr = self._fr_ls3(reorderlevel)
        else:
            # Fill rate = (E[IOH_L] - E[IOH_RL])/E[D_R]
            if eioh_l is None:
                eioh_l = self._eioh("L", reorderlevel)
            if eioh_rl is None:
                eioh_rl = self._eioh("RL", reorderlevel)
            fr = max(0., (eioh_l-eioh_rl)/(self.reviewperiod*self.mean))

        # Correction for perishables
        if self.shelflife > 0:
            # FCC >= 1?
            if self._max_oq()/(self.mean*self.shelflife) >= 1.0:
                # Large order size
                if reorderlevel <= self.leadtime*self.mean:
                    os = math.ceil(reorderlevel/self._max_oq())*self._max_oq()
                    u = min(FCC_UMAX, self.shelflife-1)
                    p_cum = 0.
                    fr = 0.
                    while u >= 0:
                        if u == 0:
                            # No EWA: we can order at t=m
                            p_order = 1.0 - p_cum
                            # Zero inventory during the lead-time
                            rloss = self.demand_dist["Mm0"].rloss(os)
                            fr += (p_order*(os-rloss)
                                   /(self.mean*(self.shelflife+self.leadtime)))
                        else:
                            p_order = (1.0
                                - self.demand_dist["Mm"
                                + str(u)].cdf(os-reorderlevel)
                                - p_cum)
                            p_cum += p_order
                            rloss = self.demand_dist["Mm" + str(u-1)].rloss(os)
                            fr += p_order*(os-rloss)/(self.mean*(self.shelflife-u+1))
                        u -= 1
                else:
                    os = math.ceil((reorderlevel-self.mean*self.leadtime)
                                   /self._max_oq())*self._max_oq()
                    fr = max(fr,
                             (os - self.demand_dist["Mm0"].rloss(os))
                             /(self.mean*self.shelflife))
        return fr

    def _ez(self, reorderlevel, acc=False):
        # Return the expected relative outdating.
        ez = 0.
        if self.shelflife > 0:
            # Determine zB
            z_b = self._ez_b(reorderlevel)
            # Get integer L and R
            lt = int(round(self.leadtime, 0))
            rp = int(round(self.reviewperiod, 0))
            if z_b >= 0.001:
                # Determine zA
                z_a = self._ez_a(reorderlevel)
                if ((not self.zreg_fifo is None)
                      and (lt in (1, 2) and rp in (1, 2))):
                    # Construct regression data key
                    sl = int(self.shelflife)
                    if sl <= 10:
                        zreg_key = "M"+str(sl)
                    elif sl <= 15:
                        zreg_key = "M15"
                    elif sl <= 20:
                        zreg_key = "M20"
                    elif sl <= 25:
                        zreg_key = "M25"
                    else:
                        zreg_key = "M30"
                    zreg_key = zreg_key + "L"+str(lt)+"R"+str(rp)
                    # Intercept
                    z_reg = self.zreg_fifo[zreg_key][0]
                    # CoV
                    z_reg += self.zreg_fifo[zreg_key][1]*math.sqrt(self.variance)/self.mean
                    # (SS+Q-1)/Mu
                    ssq = ((reorderlevel - (self.leadtime+self.reviewperiod)*self.mean)
                           + self._max_oq() - 1)/self.mean
                    if ssq > 0.:
                        z_reg += self.zreg_fifo[zreg_key][2]*ssq
                    # (Q/Mu)-R
                    qmu_r = self._max_oq()/self.mean - self.reviewperiod
                    if qmu_r > 0.:
                        z_reg += self.zreg_fifo[zreg_key][3]*qmu_r
                    # Ceil(s/Q)*Q/Mu
                    os_mu = math.ceil(reorderlevel/self._max_oq())*self._max_oq()/self.mean
                    if os_mu > 0.:
                        z_reg += self.zreg_fifo[zreg_key][4]*os_mu
                    # (1-P2) is skipped
                    # zA
                    z_reg += self.zreg_fifo[zreg_key][6]*z_a
                    # zB
                    z_reg += self.zreg_fifo[zreg_key][7]*z_b
                    # zC is skipped (only for LIFO)
                    # High shelflife?
                    if self.shelflife > 10:
                        z_reg += self.zreg_fifo[zreg_key][9]*self.shelflife
                    z_reg = max(0., z_reg)
                else:
                    z_reg = z_b
                # Correction for high waste
                ez = max(z_b, (z_reg + (z_a**4))/(1 + (z_a**3)))
            else:
                ez = max(0., z_b)
        if acc:
            return ez, VAL_APPROX
        return ez

    def _eua(self, reorderlevel, unitcap, acc=False):
        # Return the expected number of unit arrivals per review period
        # for discrete demand and lost sales.
        eua = 0.
        if reorderlevel > 0:
            if self.leadtime > self.reviewperiod:
                if acc:
                    return eua, VAL_NOTYET
                return eua
            c = max(unitcap, self.ioq)
            dist_ip_min = self._create_inv_dist(reorderlevel, "IP-")
            for ip in range(reorderlevel):
                ordersize = self._ordersize(reorderlevel, ip)
                eua += (math.ceil(ordersize/c)*dist_ip_min.apmf[ip])
            # Return only positive values
            eua = max(0., eua)
            if acc:
                return eua, self.accuracy()
            return eua
        if acc:
            return eua, VAL_EXACT
        return eua
