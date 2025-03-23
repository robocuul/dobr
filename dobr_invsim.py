# pylint: disable-msg=C0302
# -*- coding: utf-8 -*-
""" This module is designed for determining the key performance indicators
for periodic inventory control systems using discrete event simulation.

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
import math
import heapq
import copy
from operator import attrgetter
import numpy as np
from scipy.stats import t as t_test
import dobr_rsnq as dobr
import dobr_dist

# For readability, we do not always follow snake name convention
# pylint: disable-msg=C0103
# Too many instance attributes
# pylint: disable-msg=R0902
# Too few public methods
# pylint: disable-msg=R0903
# Too many return statements
# pylint: disable-msgXXX=R0911
# Too many branches
# pylint: disable-msgXXX=R0912
# Too many parameters
# pylint: disable-msgXXX=R0913
# Too many locals
# pylint: disable-msgXXX=R0914
# Too many statements
# pylint: disable-msgXXX=R0915
# pylint: disable-msgXXX=R0916

# Constants
MAXWEEKDAYS = 7             # Number of weekdays

# Event ID (number indicates priority)
DEMAND = 1
INSPECT = 2
REVIEW = 3
SUPPLY = 4
DELIVERY = 5
ENDPERIOD = 6

TWEEDIE = 1.5
SAMPLE_SIZE = 10000
MULTI_ORDER = True          # Allow more than 1 outstanding order
IOH_OUTDATE = True          # Measure IOH before the outdating

DEFAULT_SIM = {
    "targetkpi": "Fillrate",
    "targetprecision": 0.001,
    "maxrepeats": 100,
    "warmup" : 100,
    "nrperiods" : 10000,
    "poissonprocess" : False,
    "weekpattern" : False,
    "reportpmfs": True}

# PMF of IOH and IP ID's (only PMFs with values greater than -1 are collected)
NR_PMFS = 4
PMF_IOH_L = 0
PMF_IOH_RL = 1
PMF_IP_PLUS = 2
PMF_OS = 3
PMF_IP_MIN = -1
PMF_TBO = -1
PMF_W = -1

def precision(sumx, sumx2, n_val, alpha=0.05):
    """ Determine the p95 precision. """
    prec = 100.
    if n_val > 0:
        pop_var = sumx2/n_val - (sumx/n_val)**2
        # if pop_var > dobr.EPS and n_val > 1:
        if n_val > 1:
            sample_var = pop_var*n_val/(n_val-1)
            # Lookup t_(n-1,1-alpha/2) with n = repeats and alpha = 0.05
            prec = t_test.ppf(1.-alpha/2, n_val-1)*((sample_var/n_val)**0.5)
    return prec

def cycle_ref(weekday, weekpattern=False):
    """ Circular reference. """
    if weekpattern:
        weekday += 1
        if weekday >= MAXWEEKDAYS:
            weekday = 0
    return weekday




class SimResult:
    """ Class to store the simulation results. """

    def __init__(self, sku_param, report_pmfs=False):

        # Create empty KPI dictionary
        self.kpis = {}
        for kpi in dobr.KPI_CATALOGUE:
            if dobr.applicable_kpi(kpi, sku_param):
                self.kpis[kpi] = [0., 0.]
        # Init PMFs
        self.pmfs = None
        self.report_pmfs = report_pmfs
        # Determine MIOH, which determines the range of the PMFs
        self.mioh = sku_param["reorderlevel"] + max(sku_param["ioq"],
                                                    sku_param["moq"])
        if sku_param["distribution"] == "Discrete":
            self.mioh -= 1
        self.mioh = int(self.mioh)

        self.reset()

    def reset(self):
        """ Reset the results. """
        # KPIs
        for key in self.kpis:
            self.kpis[key] = [0., 0.]
        # PMFs
        if self.report_pmfs:
            self.pmfs = []
            i = 0
            while i < NR_PMFS:
                self.pmfs.append([0]*int(self.mioh+1))
                i += 1

    def get_mean(self, key):
        """ Get the kpi value. """
        return self.kpis[key][0]

    def get_variance(self, key):
        """ Get the kpi value. """
        return self.kpis[key][1]

    def put_kpi(self, sum_x, sum_x2, n_val, key):
        """ Put the kpi value. """
        mean = sum_x/n_val
        # popvar = sum_x2/n_val - mean**2
        self.kpis[key][0] = mean
        self.kpis[key][1] = sum_x2/n_val - mean**2

    def add_pmf(self, x, serie):
        """ Add the last observation to the pmf. """
        x = max(0, min(self.mioh, x))
        self.pmfs[serie][x] += 1

    def pmf_scale(self, scale):
        """ Scale the pmfs. """
        for s in range(NR_PMFS):
            if s == PMF_TBO:
                obs = sum(self.pmfs[s])
                self.pmfs[s] = [i/obs for i in self.pmfs[s]]
            else:
                self.pmfs[s] = [i*scale for i in self.pmfs[s]]


class FES:
    """ Future Event Set class. """

    def __init__(self):
        self.events = []

    def add(self, event):
        """ Add an event. """
        heapq.heappush(self.events, event)

    def next(self):
        """ Get the next event. """
        return heapq.heappop(self.events)

    def size(self):
        """ Get the number of events. """
        return len(self.events)


class EventData:
    """ Class with the event data. """

    def __init__(self, time, etype=0, item=None, data=None):
        self.time = time        # Scheduled event time
        self.etype = etype      # Type of event
        self.item = item
        self.data = data

    def __lt__(self, other):    # Compare to other events
        return (self.time < other.time
                or (self.time == other.time and self.etype < other.etype))

class DemandStream:
    """ A class to manage demand streams. """

    def __init__(self, sku_param, sim_param, weekday=0, weekfraction=None):
        self.distribution_type = sku_param["distribution"]
        self.horizon = sim_param["warmup"] + sim_param["nrperiods"]
        self.weekday = weekday
        # Period demand
        self.mean = sku_param["mean_perioddemand"]
        # Assumption: Constant VTM
        vtm = (sku_param["stdev_perioddemand"]**2)/self.mean
        # Update demand?
        if weekfraction is None:
            self.week_pattern = False
        else:
            self.week_pattern = True
            self.mean *= weekfraction * MAXWEEKDAYS

        self.iat_random = 0
        self.grab_random = 0

        # Lead-time distribution? Not dependent on mean demand
        self.lt_random = 0
        if sku_param["leadtime"] > dobr.EPS and sku_param["stdev_leadtime"] > dobr.EPS:
            self.lt_random = 1
            self.stream_lt = None

        if self.mean > dobr.EPS:
            self.stream_demand = None
            # Random grabbing?
            self.lifo = 1.0 - sku_param["fifo"]
            if self.lifo > 0. and self.lifo < 1.:
                self.grab_random = 1
                self.stream_grab = None
            # Poisson process?
            if sim_param["poissonprocess"]:
                self.iat_random = 1
                self.stream_iat = None
                if self.distribution_type == "Discrete":
                    prob = 2.0/(1+vtm)
                    l_rate = self.mean*prob
                    self.dist_os = prob
                    self.dist_iat = l_rate
                else:
                    # Only Gamma with Tweedie
                    dispersion = vtm/(self.mean**(TWEEDIE-1))
                    l_rate = ((self.mean**(2-TWEEDIE))
                                  /((2.0-TWEEDIE)*dispersion))
                    shape = (2.0-TWEEDIE)/(TWEEDIE-1.0)
                    scale = ((TWEEDIE-1.)*dispersion
                                 /(self.mean**(1.0-TWEEDIE)))
                    self.dist_os = [shape, scale]

            else:
                # Demand order size parameters
                if self.distribution_type == "Discrete":
                    self.dist_os = dobr_dist.select_dist(self.mean, vtm)
                    # Add the ppf array for this distribution
                    self.dist_os.create_ppf_array()
                elif self.distribution_type == "Gamma":
                    # With Mu = Alpha*Theta and Var = Alpha*Theta^2:
                    # Shape Alpha = (Mu/Sigma)^2 = Mu/VTM
                    shape = self.mean/vtm
                    # Scale Theta = Mu/Alpha = VTM
                    self.dist_os = [shape, vtm]
                elif self.distribution_type == "Normal":
                    stdev = (vtm*self.mean)**0.5
                    self.dist_os = [self.mean, stdev]



        # Leadtime variation
        if self.lt_random == 1:
            if self.iat_random == 0:
                # Only discrete values for the lead-time
                lt_vtm = (sku_param["stdev_leadtime"]**2)/sku_param["leadtime"]
                self.dist_lt = dobr_dist.select_dist(sku_param["leadtime"], lt_vtm)
                # Add the ppf array for this distribution
                self.dist_lt.create_ppf_array()
            else:
                # We assume a Gamma distributed leadtime in case of a poisson process
                # With Mu = Alpha*Theta and Var = Alpha*Theta^2:
                # Shape Alpha = (Mu/Sigma)^2
                shape = (sku_param["leadtime"]/sku_param["stdev_leadtime"])**2
                # Scale Theta = Mu/Alpha
                scale = sku_param["leadtime"]/shape
                self.dist_lt = [shape, scale]

    def reset_streams(self, repeat):
        """Reset the demand stream. """
        # Sample size = horizon
        if self.week_pattern:
            sample_size = int(self.horizon/MAXWEEKDAYS + MAXWEEKDAYS)
        else:
            sample_size = self.horizon
        # Determine stream number
        stream_nr = (repeat*(self.weekday+1)
                     *(self.iat_random+self.grab_random+self.lt_random+1))
        if self.iat_random == 1:
            # IAT = Inter-Arrival Time
            self.stream_iat = ExpoDraw(stream_nr,
                sample=sample_size, scale=1.0/self.dist_iat)
            # Demand order size
            if self.distribution_type == "Discrete":
                self.stream_demand = ShiftGeoDraw(stream_nr,
                    sample=sample_size, p=self.dist_os)
            else:
                self.stream_demand = GammaDraw(stream_nr,
                    sample=sample_size, shape=self.dist_os[0],
                    scale=self.dist_os[1])
        else:
            # Demand order size
            if self.distribution_type == "Discrete":
                self.stream_demand = DiscreteDraw(stream_nr,
                    sample=sample_size, dist=self.dist_os)
            elif self.distribution_type == "Gamma":
                # Use pre-calculated shape and scale
                self.stream_demand = GammaDraw(stream_nr,
                    sample=sample_size, shape=self.dist_os[0], scale=self.dist_os[1])
            elif self.distribution_type == "Normal":
                # Use pre-calculated loc and scale
                self.stream_demand = NormalDraw(stream_nr,
                    sample=sample_size, loc=self.dist_os[0], scale=self.dist_os[1])
        # Grabbing
        if self.grab_random == 1:
            stream_nr += 1
            if self.distribution_type == "Discrete":
                # Amount of grabbing is determined with a binomial
                # distribution (repeat trials)
                self.stream_grab = FlipDraw(stream_nr,
                                            sample=sample_size,
                                            p=self.lifo)
            else:
                # Amount of grabbing is determined with a triangular
                # distribution
                self.stream_grab = TriangularDraw(stream_nr,
                                                  sample=sample_size,
                                                  mode=self.lifo)
        # Leadtime variance
        if self.lt_random == 1:
            stream_nr += 1
            if self.iat_random == 0:
                # Discrete distributed leadtime
                self.stream_lt = DiscreteDraw(stream_nr,
                                              sample=sample_size,
                                              dist=self.dist_lt)
            else:
                # Gamma distributed leadtime
                self.stream_lt = GammaDraw(stream_nr,
                                           sample=sample_size,
                                           shape=self.dist_lt[0],
                                           scale=self.dist_lt[1])

    def gen_next(self, current_time, first=False):
        """ Generate next demand. """
        # Note: only called if mean is greater than zero
        iat = 0.
        demand = 0.
        no_demand = True
        while no_demand and current_time + iat < self.horizon:
            if self.iat_random == 0:
                # IAT advances with integer periods
                if self.week_pattern:
                    iat += MAXWEEKDAYS
                else:
                    iat += 1.
            else:
                # Poisson process
                iat += self.stream_iat.rvs()
                if self.week_pattern:
                    if first:
                        # Put the first arrival halfway in the period
                        iat = 0.5
                    else:
                        # Determine delta between the current and the next period
                        delta = int(current_time + iat) - int(current_time)
                        # Add a week (minus the period length = 1 day)
                        # for each period difference
                        iat += delta*(MAXWEEKDAYS - 1)
            # Weekday stays the same
            demand = self.stream_demand.rvs()
            if demand != 0. or first:
                no_demand = False

        # Grabbing?
        demand_lifo = self.lifo_demand(demand)
        return iat, demand, demand_lifo

    def lifo_demand(self, demand):
        """ Returns the grabbing (LIFO) part of the demand. """
        if self.lifo == 0:
            # 100% FIFO, no LIFO
            return 0
        if self.lifo == 1:
            # 0% FIFO, only LIFO
            return demand
        # Mixed FIFO-LIFO
        if self.distribution_type == "Discrete":
            # Amount of grabbing is determined with a binomial
            # distribution (repeat trials)
            return self.stream_grab.rvs(t=int(demand))
        # Amount of grabbing is determined with a triangular
        # distribution
        return demand*self.stream_grab.rvs()


class DistDraw:
    """ A class used to generate uniform variates in bulk. """

    def __init__(self, seed, sample=SAMPLE_SIZE):
        self.stream = np.random.default_rng(seed)
        self.sample_size = sample
        self.draws = None
        self.idx = -1

    def _resample(self):
        self.draws = self.stream.uniform(0.0, 1.0, size=self.sample_size)
        self.idx = 0

    def rvs(self, t=1):
        """ Draw new random variates. """
        if self.idx == -1 or self.idx + t > self.sample_size:
            self._resample()
        u = self.draws[self.idx]
        self.idx += 1
        return u


class ShiftGeoDraw(DistDraw):
    """ A class used to generate Shifted Geometric variates in bulk. """

    def __init__(self, seed, sample=SAMPLE_SIZE, p=1):
        super().__init__(seed, sample=sample)
        self.p = p

    def rvs(self, t=1):
        """ Draw new random variates. """
        if self.p < 1.0:
            if self.idx == -1 or self.idx + t > self.sample_size:
                self._resample()
            u = self.draws[self.idx]
            x = 1 + int(math.log(u)/math.log(1-self.p))
            self.idx += 1
        else:
            x = 1
        return x


class TriangularDraw(DistDraw):
    """ A class used to generate Triangular variates in bulk. """

    def __init__(self, seed, sample=SAMPLE_SIZE, mode=0.5):
        super().__init__(seed, sample=sample)
        self.mode = mode

    def rvs(self, t=1):
        """ Draw new random variates. """
        if self.idx == -1 or self.idx + t > self.sample_size:
            self._resample()
        u = self.draws[self.idx]
        if u <= self.mode:
            x = math.sqrt(u*self.mode)
        else:
            x = 1.0-math.sqrt((1.0-self.mode)*(1.0-u))
        self.idx += 1
        return x


class ExpoDraw(DistDraw):
    """ A class used to generate Exponential variates in bulk. """

    def __init__(self, seed, sample=SAMPLE_SIZE, scale=1.):
        super().__init__(seed, sample=sample)
        self.scale = scale              # Beta = 1/Lambda

    def _resample(self):
        self.draws = self.stream.exponential(scale=self.scale,
                                             size=self.sample_size)
        self.idx = 0


class GammaDraw(DistDraw):
    """ A class used to generate Gamma variates in bulk. """

    def __init__(self, seed, sample=SAMPLE_SIZE, shape=1, scale=1):
        super().__init__(seed, sample=sample)
        self.shape = shape          # Alpha
        self.scale = scale          # Theta

    def _resample(self):
        self.draws = self.stream.gamma(self.shape, scale=self.scale,
                                       size=self.sample_size)
        self.idx = 0


class NormalDraw(DistDraw):
    """ A class used to generate Normal variates in bulk. """

    def __init__(self, seed, sample=SAMPLE_SIZE, loc=0, scale=1):
        super().__init__(seed, sample=sample)
        self.loc = loc
        self.scale = scale

    def _resample(self):
        self.draws = self.stream.normal(loc=self.loc, scale=self.scale,
                                       size=self.sample_size)
        self.idx = 0


class DiscreteDraw(DistDraw):
    """ A class used to generate mixed discrete variates in bulk. """

    def __init__(self, seed, sample=SAMPLE_SIZE, dist=None):
        super().__init__(seed, sample=sample)
        self.dist = dist

    def rvs(self, t=1):
        if self.idx == -1 or self.idx + t > self.sample_size:
            self._resample()
        u = self.draws[self.idx]
        if not self.dist is None:
            u = self.dist.getvariateppf(u)
        self.idx += 1
        return u


class FlipDraw(DistDraw):
    """ A class used to generate Bernoulli or Binomial variates in bulk. """

    def __init__(self, seed, sample=SAMPLE_SIZE, p=0.5):
        super().__init__(seed, sample=sample)
        self.p = p

    def rvs(self, t=1):
        if self.idx == -1 or self.idx + t > self.sample_size:
            self._resample()
        heads = 0
        flip = 0
        while flip < t:
        # for flip in range(t):
            if self.draws[self.idx] < self.p:
                heads += 1
            self.idx += 1
            flip += 1
        return heads


class BatchStatus:
    """A class used to track the status of a batch."""

    def __init__(self, available, ordered, expiration=-1):
        self.available = available      # Time when the batch becomes available
        self.expiration = expiration    # Time when the batch expires
        self.ordered = ordered          # Ordered amount
        self.delivered = 0              # Delivered amount
        self.onhand = 0                 # Total amount on-hand
        self.backroom = 0               # Amount in the backroom

    def __str__(self):
        line = f"Timings: {self.available: 8.2f} <= {self.expiration: 8.2f}"
        line = (line + f" (Order {self.ordered: 5.1f};"
            + f" On-hand {self.onhand: 5.1f}; Backroom {self.backroom: 5.1f})")
        return line

    def expired(self, ref_time):
        """ Check if a perishable batch is expired. """
        return bool(self.expiration > 0 and self.available < ref_time
              and self.expiration <= ref_time)

    def depleted(self, ref_time):
        """ Check if a non-perishable batch is depleted. """
        return bool(self.expiration < 0 and self.available < ref_time
              and self.onhand < dobr.EPS)

    def imminent(self, ref_time):
        """ Check if a batch is imminent to arrive. """
        return bool(abs(self.available - ref_time) < dobr.EPS)

    def sojourn_time(self, ref_time):
        """ Determine the sojourn time of a batch. """
        return max(0., ref_time-self.available)


class BatchList:
    """A class used to operate on batches."""

    def __init__(self):
        self.book = []

    def append(self, batch, perish=False):
        """ Append a batch to the end of the batch list. """
        self.book.append(batch)
        nr_batches = len(self.book)
        if nr_batches > 1:
            # Check time order of the last appended batch
            if perish:
                if (self.book[nr_batches-2].expiration
                    > self.book[nr_batches-1].expiration):
                    # Sortation required
                    self.book.sort(key=attrgetter("expiration"))
            else:
                if (self.book[nr_batches-2].available
                    > self.book[nr_batches-1].available):
                    # Sortation required
                    self.book.sort(key=attrgetter("available"))

    def retrieval(self, sales, ref_time, fifo=True, backroom=False):
        """ Retrieve items from the batches in the list. """
        nr_batches = len(self.book)
        if nr_batches == 0:
            print("DoBr: no batches to retrieve "+str(ref_time))
            return -1.
        if fifo:
            # Start with oldest (=first) batch in the list
            batchnr = 1
        else:
            # Start with newest (=last) batch in the list
            batchnr = nr_batches

        retrieval = 0.
        shelflife_sales = 0.
        while retrieval < sales and (1 <= batchnr <= nr_batches):
            batch = self.book[batchnr-1]
            if batch.available < ref_time:
                if backroom:
                    batch_retrieval = min(batch.backroom,
                            sales - retrieval)
                else:
                    batch_retrieval = min(batch.onhand - batch.backroom,
                            sales - retrieval)
                if batch_retrieval > dobr.EPS:
                    retrieval += batch_retrieval
                    batch.onhand -= batch_retrieval
                    if backroom:
                        batch.backroom -= batch_retrieval
                    shelflife_sales += (batch_retrieval
                            *(ref_time-batch.available))

            if fifo:
                batchnr += 1
            else:
                batchnr -= 1

        return shelflife_sales

    def clean(self, ref_time):
        """ Clean the batch list. """
        outdated_onhand = 0.
        outdated_backroom = 0.
        # Start with the last for performance reasons
        batchnr = len(self.book)
        while batchnr > 0:
            batch = self.book[batchnr-1]
            if batch.expired(ref_time) or batch.depleted(ref_time):
                # Remove from the list
                outdated_onhand += batch.onhand
                outdated_backroom += batch.backroom
                self.book.pop(batchnr-1)
            batchnr -= 1
        return outdated_onhand, outdated_backroom

    def outstanding(self, ref_time):
        """ Determine the number of outstanding batches of the batch list. """
        noo = 0
        nr_batches = len(self.book)
        batchnr = 1
        while batchnr <= nr_batches:
            batch = self.book[batchnr-1]
            if batch.available >= ref_time and batch.ordered > 0:
                noo += 1
            batchnr += 1
        return noo

    def onhand(self, ref_time, unitcap=1):
        """ Determine the number of on-hand batches of the batch list. """
        noh = 0
        nr_loc = 0
        nr_batches = len(self.book)
        batchnr = 1
        while batchnr <= nr_batches:
            batch = self.book[batchnr-1]
            if batch.available <= ref_time and batch.onhand > 0:
                noh += 1
                nr_loc += math.ceil(batch.onhand/unitcap)
            batchnr += 1
        return noh, nr_loc

    def deliver(self, ref_time):
        """ Which batch is delivered? """
        # Used in EWA
        nr_batches = len(self.book)
        delivered = 0
        batchnr = 1
        while batchnr <= nr_batches:
            batch = self.book[batchnr-1]
            if batch.imminent(ref_time):
                # Deliver to stock
                batch.onhand = batch.ordered
                delivered += batch.ordered
            batchnr += 1
        return delivered

    def move_from_backroom(self, replenishment, ref_time):
        """ Move batches from the backroom to the shelves. """
        nr_batches = len(self.book)
        # Start with oldest (=first) batch in the list
        batchnr = 1
        movement = 0.
        while movement < replenishment and (batchnr <= nr_batches):
            batch = self.book[batchnr - 1]
            if batch.available < ref_time:
                batch_movement = min(batch.backroom,
                    replenishment - movement)
                if batch_movement > dobr.EPS:
                    movement += batch_movement
                    batch.backroom -= batch_movement
            batchnr += 1


class InventoryStatus:
    """ Class used to track the status of the inventory. """

    def __init__(self):
        # Inventory status
        self.ip = 0.            # Inventory position
        self.backorders = 0.    # Amount of backorders
        self.onhand = 0.        # Amount on-hand
        self.backroom = 0.      # Amount in the backroom
        # Timing status
        self.previnvchange = 0. # Previous inventory change
        self.batches = None     # Batch list

    def reset_inventory(self, reorderlevel, shelflife):
        """ Reset the inventory status. """
        # Current inventory status
        self.ip = reorderlevel
        self.backorders = 0.
        self.onhand = self.ip
        self.backroom = 0.
        # Create a (new) batch list
        self.batches = BatchList()
        # Add the current inventory as first batch
        if shelflife > 0:
            first_batch = BatchStatus(0., self.onhand,
                                   expiration=shelflife)
        else:
            first_batch = BatchStatus(0., self.onhand)
        first_batch.onhand = first_batch.ordered
        self.batches.append(first_batch)

        # Timing status
        self.previnvchange = 0.         # Previous inventory change

    def shelf(self):
        """ Shelf inventory = On-hand inventory minus backroom inventory. """
        return self.onhand-self.backroom

class ReviewPeriodStatus:
    """ Class to track changes in a review period. """

    def __init__(self):
        # Review period stats
        self.demand = 0.
        self.outdate = 0.
        self.nr_conc = 0

    def reset_review_period(self):
        """ Reset the status attributes. """
        self.demand = 0.
        self.outdate = 0.
        self.nr_conc = 0

class StockPoint:
    """ Class to describe a stock point. """

    def __init__(self, sku_param, sim_param, weekfractions):
        # SKU parameters
        self.sku_param = sku_param
        # Simulation parameters:
        self.sim_param = sim_param

        # Create result structure
        self.sim_res = SimResult(sku_param, report_pmfs=self.sim_param["reportpmfs"])

        # SKU (= item) parameters
        # Adjust MOQ
        if sku_param["moq"] == 0:
            self.sku_param["moq"] = self.sku_param["ioq"]
        # Adjust tranfer unit capacity
        if sku_param["unitcap"] == 0:
            self.sku_param["unitcap"] = max(1, self.sku_param["ioq"])

        self.demand = []
        self.rol = []
        if self.sim_param["weekpattern"]:
            self.rol = self._set_rol(sku_param, weekfractions)
            # self.rol = [sku_param["reorderlevel"]] * MAXWEEKDAYS
            for weekday in range(MAXWEEKDAYS):
                self.demand.append(DemandStream(sku_param, sim_param,
                                                weekday=weekday,
                                                weekfraction=weekfractions[weekday]))
        else:
            self.demand.append(DemandStream(sku_param, sim_param))
            self.rol.append(sku_param["reorderlevel"])

        # Statistics
        self.stats_sum = {}
        for kpi in self.sim_res.kpis:
            self.stats_sum[kpi] = [0., 0.]
        # Add Demand and Sales to the dictionary
        self.stats_sum["Demand"] = [0., 0.]
        self.stats_sum["Sales"] = [0., 0.]

        # Inventory
        self.inv = InventoryStatus()
        # Review period stats
        self.rp = ReviewPeriodStatus()
        # Order timings
        self.prev_order_time = 0
        self.order_times = []

    def reset_repeat(self, repeat):
        """ Reset the data for the new repeat. """
        # Reset inventory status
        self.inv.reset_inventory(self.sku_param["reorderlevel"],
                                 self.sku_param["shelflife"])

        # Reset order timings
        self.prev_order_time = 0        # To determine the TBO
        self.order_times = []           # To determine ELT

        # Reset statistics
        for key in self.stats_sum:
            self.stats_sum[key][0] = 0.
            self.stats_sum[key][1] = 0.

        # Init random streams
        if self.sim_param["weekpattern"]:
            for weekday in range(MAXWEEKDAYS):
                if self.demand[weekday].mean > dobr.EPS:
                    self.demand[weekday].reset_streams(repeat)
        else:
            self.demand[0].reset_streams(repeat)

    def _set_rol(self, sku_param, weekfractions):
        """ Determine the reorder levels per weekday. """
        rol_list = []

        # First, find the peak R+L demand
        mean_week = sku_param["mean_perioddemand"] * MAXWEEKDAYS
        peak_day = 0
        peak_frac_rl = 0.
        for weekday in range(MAXWEEKDAYS):
            frac_rl = 0.
            t = weekday
            for _ in range(int(sku_param["reviewperiod"]
                                + sku_param["leadtime"])):
                t = cycle_ref(t, weekpattern=self.sim_param["weekpattern"])
                frac_rl += weekfractions[t]
            if peak_day == 0 or frac_rl > peak_frac_rl:
                peak_day = weekday
                peak_frac_rl = frac_rl
        # Safety stock is based on the peak
        ss = sku_param["reorderlevel"] - peak_frac_rl*mean_week
        for weekday in range(MAXWEEKDAYS):
            frac_rl = 0.
            t = weekday
            for _ in range(int(sku_param["reviewperiod"]
                                + sku_param["leadtime"])):
                t = cycle_ref(t, weekpattern=self.sim_param["weekpattern"])
                frac_rl += weekfractions[t]
            rol_day = max(1, int(math.ceil(ss + frac_rl*mean_week)))
            rol_list.append(rol_day)
        return rol_list

    def _add_sumstat(self, stat, value):
        if stat in self.stats_sum:
            # Add the passed value to the statistic stat
            self.stats_sum[stat][0] += value
            self.stats_sum[stat][1] += value**2

    def subtract_sumstat(self, stat, value):
        """ Subtract the passed value to the statistic stat. """
        if stat in self.stats_sum:
            self.stats_sum[stat][0] -= value
            self.stats_sum[stat][1] -= value**2

    def event_sales(self, current_time, demand_data):
        """ Execute the sales process event. """

        # Unpack demand tuple
        demand = demand_data[0]
        demand_lifo = demand_data[1]
        self.rp.demand += demand
        # Status before demand
        prev_onhand = self.inv.onhand

        sales = 0.
        shelflife_sales = 0.
        if demand < 0.:
            # Returned demand (only with Normal distribution)
            sales = demand
            self.inv.ip -= sales
            cleared_backlog = min(self.inv.backorders, -demand)
            if cleared_backlog > 0.:
                self.inv.backorders -= cleared_backlog
                demand += cleared_backlog
            if demand < 0.:
                self.inv.onhand -= demand
                if len(self.inv.batches.book) > 0.:
                    self.inv.batches.book[0].onhand -= demand
        elif demand > 0.:
            # First, sell from the shelf
            if self.inv.shelf() > dobr.EPS:
                # We have inventory on the shelf
                sales_shelf = min(self.inv.shelf(), demand)
                # LIFO before FIFO
                sales_lifo = min(sales_shelf, demand_lifo)
                if sales_lifo > dobr.EPS:
                    shelflife_sales += self.inv.batches.retrieval(sales_lifo,
                        current_time, fifo=False, backroom=False)
                # Remainder of the sales is retrieved FIFO
                sales_fifo = sales_shelf - sales_lifo
                if sales_fifo > dobr.EPS:
                    shelflife_sales += self.inv.batches.retrieval(sales_fifo,
                            current_time, fifo=True, backroom=False)
                sales += sales_shelf
                self.inv.onhand -= sales_shelf
                self.inv.ip -= sales_shelf
            # Second, sell from the backroom
            if (demand - sales > dobr.EPS and self.inv.backroom > dobr.EPS
                  and self.sku_param["concurrent"]):
                # Sales are limited by the backroom inventory
                sales_backroom = min(self.inv.backroom, demand - sales)
                sales += sales_backroom
                # Replenishments are in multiples of the shelf capacity
                replenishment = min(self.inv.backroom,
                                    math.ceil(sales_backroom/self.sku_param["shelfspace"])
                                    *self.sku_param["shelfspace"])
                self.rp.nr_conc += math.ceil(replenishment/self.sku_param["shelfspace"])
                # Always FIFO retrieval from the backroom
                shelflife_sales += self.inv.batches.retrieval(sales_backroom,
                    current_time, fifo=True, backroom=True)
                # Move the remaining part also in the batches
                self.inv.batches.move_from_backroom(replenishment - sales_backroom,
                    current_time)
                # Update inventory stats
                self.inv.onhand -= sales_backroom
                self.inv.ip -= sales_backroom
                self.inv.backroom -= replenishment
            # Remaining demand is lost sales or backordered
            if demand - sales > dobr.EPS and (not self.sku_param["lostsales"]):
                # In case of a backroom and no concurrent replenishments
                # we can have on-hand inventory AND backorders
                self.inv.backorders += (demand - sales)
                self.inv.ip -= (demand - sales)

        # Register
        if current_time >= self.sim_param["warmup"]:
            self._add_sumstat("Sales", sales)
            self._add_sumstat("EST", shelflife_sales )
            # Integral inventory measurement
            delta_time = current_time-self.inv.previnvchange
            if self.sim_param["poissonprocess"]:
                self._add_sumstat("EIOH_Cont", delta_time*prev_onhand)
            else:
                self._add_sumstat("EIOH_Cont", delta_time
                    *(delta_time*prev_onhand+self.inv.onhand)/(delta_time+1))
        # Store time of last inventory change
        self.inv.previnvchange = current_time

    def event_inspect(self, current_time):
        """ Execute the inspection process event. """
        outdated_onhand, outdated_backroom = self.inv.batches.clean(current_time)
        self.inv.onhand -= outdated_onhand
        self.inv.backroom -= outdated_backroom
        self.inv.ip -= outdated_onhand
        self.rp.outdate += outdated_onhand
        if current_time >= self.sim_param["warmup"]:
            self._add_sumstat("EW", outdated_onhand)

    def event_review(self, current_time, weekday):
        """ Execute the review process event. """
        noo = self.inv.batches.outstanding(current_time)

        # Register
        if current_time >= self.sim_param["warmup"]:
            # Inital inventory is also delivered
            if self.stats_sum["ESUP"][0] < dobr.EPS:
                self._add_sumstat("ESUP", self.inv.onhand)
            # Distribution IP before ordering
            if self.sim_res.report_pmfs and PMF_IP_MIN > -1:
                self.sim_res.add_pmf(int(self.inv.ip), PMF_IP_MIN)

        # Initial, the order size is zero
        ordersize = 0.
        # Can we order?
        if MULTI_ORDER or noo <= 1:

            rol = self.rol[weekday]

            # Apply EWA?
            if (self.inv.ip > 0 and self.sku_param["shelflife"] > 0 and self.sku_param["EWA"]
                    and self.sku_param["leadtime"]+self.sku_param["reviewperiod"] > 1):
                # Create a deep copy of the current inventory status
                # for the projected (prj_) status
                prj_onhand = self.inv.onhand
                prj_backroom = self.inv.backroom
                prj_ip = self.inv.ip
                # We replaced the standard deepcopy operation
                #       prj_batches = copy.deepcopy(self.inv.batches)
                # with a manual deep copy for performance reasons
                prj_batches = BatchList()
                for i in range(len(self.inv.batches.book)):
                    prj_batches.append(copy.copy(self.inv.batches.book[i]))

                lr_1 = self.sku_param["leadtime"]+self.sku_param["reviewperiod"]-1
                carry_demand = 0.
                prj_p = 1
                if self.sim_param["weekpattern"]:
                    prj_weekday = cycle_ref(weekday)
                else:
                    prj_weekday = 0
                est_outdating = 0
                est_shelflife_sales = 0
                while prj_p <= int(math.ceil(lr_1)):
                    # Delivery of batches in the pipeline
                    if prj_ip > prj_onhand:
                        delivered = prj_batches.deliver(current_time+prj_p)
                        prj_onhand += delivered
                    # Demand
                    if prj_p > lr_1:
                        frac_p = lr_1-int(lr_1)
                    else:
                        frac_p = 1.
                    # EWA requires integer demands per period
                    est_demand = int(round(frac_p*self.demand[prj_weekday].mean
                                           + carry_demand + dobr.EPS, 0))
                    carry_demand = self.demand[prj_weekday].mean - est_demand
                    # Sales
                    est_sales = min(est_demand, prj_onhand)
                    # FIFO and LIFO
                    est_sales_fifo = int(round(
                            self.sku_param["fifo"]*est_sales + dobr.EPS, 0))
                    est_sales_lifo = est_sales - est_sales_fifo
                    if est_sales_lifo > dobr.EPS:
                        est_shelflife_sales += prj_batches.retrieval(est_sales_lifo,
                                current_time+prj_p,
                                fifo=False, backroom=False)
                    # Remainder of the sales is retrieved FIFO
                    if est_sales_fifo > dobr.EPS:
                        est_shelflife_sales += prj_batches.retrieval(est_sales_fifo,
                                current_time+prj_p,
                                fifo=True, backroom=False)
                    # Outdating
                    est_out_onhand, est_out_backroom = prj_batches.clean(current_time+prj_p)
                    # Update inventory status
                    prj_onhand -= (est_sales + est_out_onhand)
                    prj_backroom -= (est_sales + est_out_backroom)
                    prj_ip -= (est_sales + est_out_onhand)
                    # Cummulative outdating
                    est_outdating += est_out_onhand
                    # Next projected period
                    prj_p += 1
                    if self.sim_param["weekpattern"]:
                        prj_weekday = cycle_ref(prj_weekday)

                mod_ip = self.inv.ip - est_outdating
            else:
                mod_ip = self.inv.ip

            undershoot = rol - mod_ip
            if undershoot > dobr.EPS:
                if self.sku_param["distribution"] == "Discrete":
                    ordersize = int(
                        (undershoot-1+self.sku_param["moq"])
                        /self.sku_param["ioq"])*self.sku_param["ioq"]
                else:
                    if self.sku_param["ioq"] > dobr.EPS:
                        ordersize = int(
                            (undershoot+self.sku_param["moq"])
                            /self.sku_param["ioq"])*self.sku_param["ioq"]
                    else:
                        ordersize = undershoot

        # Delivery moment (also for empty orders)
        delivery_moment = current_time
        if self.demand[weekday].lt_random == 1:
            # Stochastic lead-time
            delivery_moment += self.demand[weekday].stream_lt.rvs()
        else:
            # Deterministic or zero lead-time
            delivery_moment += self.sku_param["leadtime"]

        # Positive order size?
        if ordersize > dobr.EPS:
            if self.sku_param["lostsales"] and self.sku_param["ROS"]:
                # Bijvank & Johansen: limit to s*R/(L+R)
                max_os = math.ceil(rol*self.sku_param["reviewperiod"]
                    /(self.sku_param["leadtime"]+self.sku_param["reviewperiod"]))
                # Respect the MOQ
                max_os = max(self.sku_param["moq"], max_os)
                # Round to the nearest IOQ
                max_os = round(max_os/self.sku_param["ioq"])*self.sku_param["ioq"]
                # Limit reached?
                ordersize = min(ordersize, max_os)
            self.inv.ip += ordersize
            # TBO = Time Between Orders
            tbo = current_time - self.prev_order_time
            self.prev_order_time = current_time
            # ELT: keep track of the oldest order
            self.order_times.append(current_time)
            if current_time >= self.sim_param["warmup"]:
                self._add_sumstat("EOL", 1.)
                self._add_sumstat("EOS", ordersize)
                self._add_sumstat("EUA", math.ceil(ordersize/self.sku_param["unitcap"]))
                if self.sim_res.report_pmfs:
                    if PMF_OS > -1:
                        if self.sku_param["ioq"] > dobr.EPS:
                            self.sim_res.add_pmf(int(ordersize
                                                     /self.sku_param["ioq"]), PMF_OS)
                        else:
                            self.sim_res.add_pmf(int(round(ordersize, 0)), PMF_OS)
                    if PMF_TBO > -1:
                        self.sim_res.add_pmf(int(tbo), PMF_TBO)

        # Distribution IP after ordering
        if (self.sim_res.report_pmfs and
              current_time >= self.sim_param["warmup"]):
            if PMF_IP_PLUS > -1:
                self.sim_res.add_pmf(int(self.inv.ip), PMF_IP_PLUS)
            self._add_sumstat("NOO", noo)

        # Add order to batch list
        if self.sku_param["shelflife"] > 0:
            batch = BatchStatus(
                delivery_moment, ordersize,
                expiration=delivery_moment+self.sku_param["shelflife"])
            self.inv.batches.append(batch, perish=True)
        else:
            batch = BatchStatus(delivery_moment, ordersize)
            self.inv.batches.append(batch, perish=False)

        return batch

    def event_idle(self, current_time):
        """ Event is only invoked if we have a positive shelf space. """

        # Did we have concurrent replenishments to report?
        if self.sku_param["concurrent"]:
            if current_time >= self.sim_param["warmup"]:
                if self.rp.nr_conc > dobr.EPS:
                    self._add_sumstat("ENCR", self.rp.nr_conc)
                    #self._add_sumstat("ENCR", 1.)
        # Do we have backroom inventory?
        if self.inv.backroom > 0:
            # Is there a backlog?
            if self.inv.backorders > dobr.EPS:
                # First clear the backlog
                cleared_backlog = min(self.inv.backorders, self.inv.backroom)
                self.inv.backroom -= cleared_backlog
                self.inv.backorders -= cleared_backlog
                self.inv.onhand -= cleared_backlog
                # IP stays the same
                # Always clear backlog FIFO
                self.inv.batches.retrieval(cleared_backlog, current_time,
                    fifo=True, backroom=True)

            # Fill the shelf
            replenishment = min(self.inv.backroom, self.sku_param["shelfspace"] - self.inv.shelf())
            self.inv.backroom -= replenishment
            self.inv.batches.move_from_backroom(replenishment, current_time)

            if current_time >= self.sim_param["warmup"]:
                if replenishment > dobr.EPS:
                    self._add_sumstat("ENIR", 1. )

    def event_delivery(self, current_time, delivery_batch):
        """ Execute the delivery process event. """

        delivery = delivery_batch.ordered
        if delivery > dobr.EPS:
            # ELT: determine the oldest order
            oldest_order_time = self.order_times.pop(0)

        # Stats just before a potential delivery (R+L)
        if current_time >= self.sim_param["warmup"]:
            ioh = self.inv.onhand
            shelf = self.inv.shelf()
            if IOH_OUTDATE:
                ioh += self.rp.outdate
                shelf += self.rp.outdate
            self._add_sumstat("ESUP", delivery)
            self._add_sumstat("EIOH_RL", ioh)
            self._add_sumstat("EBO_RL", self.inv.backorders)
            if shelf > dobr.EPS:
                self._add_sumstat("Readyrate", 1. )
            # Integral inventory measurement
            delta_time = current_time-self.inv.previnvchange
            self._add_sumstat("EIOH_Cont", ioh*delta_time)
            # Effective lead-time
            if delivery > dobr.EPS:
                elt = current_time - oldest_order_time
                self._add_sumstat("ELT", elt)

            # Distribution IOH before potential delivery (R+L)
            if self.sim_res.report_pmfs and PMF_IOH_RL > -1:
                self.sim_res.add_pmf(int(ioh), PMF_IOH_RL)

        # Store time of last inventory change
        self.inv.previnvchange = current_time

        # Positive delivery?
        cleared_backlog = 0
        if delivery > dobr.EPS:
            # Backorders?
            if (not self.sku_param["lostsales"]) and self.inv.backorders > dobr.EPS:
                # Uncorrected batch and location data
                noh = 1
                nr_loc = math.ceil(delivery/self.sku_param["unitcap"])
                # Clear the backlog
                cleared_backlog = min(self.inv.backorders, delivery)
                delivery -= cleared_backlog
                self.inv.backorders -= cleared_backlog

            # Do we still have something to put in the inventory?
            if delivery > dobr.EPS:
                # In case of a limited shelf space,
                # move the surplus in the backroom
                if self.sku_param["shelfspace"] > 0:
                    surplus = max(0, self.inv.shelf() + delivery
                        - self.sku_param["shelfspace"])
                    self.inv.backroom += surplus
                    delivery_batch.backroom = surplus
                self.inv.onhand += delivery
                delivery_batch.onhand = delivery

                # IP stays the same,
                # has been corrected during allocation if necessary

        # Stats: After potential delivery
        if current_time >= self.sim_param["warmup"]:
            ioh = self.inv.onhand
            if IOH_OUTDATE:
                ioh += self.rp.outdate

            self._add_sumstat("EIOH_L", ioh)
            self._add_sumstat("EBO_L", self.inv.backorders)
            if cleared_backlog < dobr.EPS:
                noh, nr_loc = self.inv.batches.onhand(current_time,
                                             self.sku_param["unitcap"])
            self._add_sumstat("ENB", noh)
            self._add_sumstat("EUSL_L", nr_loc)

            # Distribution IOH after potential delivery (L)
            if self.sim_res.report_pmfs and PMF_IOH_L > -1:
                self.sim_res.add_pmf(int(ioh), PMF_IOH_L)

            # Backroom?
            if self.inv.backroom > 0.:
                self._add_sumstat("POC", 1. )
                self._add_sumstat("EIBR", self.inv.backroom )

    def event_endperiod(self, current_time):
        """ Execute the end of period event. """
        if current_time >= self.sim_param["warmup"]:
            # Register period demand
            self._add_sumstat("Demand", self.rp.demand)
            if self.sim_res.report_pmfs and PMF_W > -1 and self.rp.outdate > 0.:
                # Register period outdating
                self.sim_res.add_pmf(int(self.rp.outdate), PMF_W)
        # Reset review period stats
        self.rp.reset_review_period()

    def calc_repeatkpis(self, actual_nrperiods):
        """ Calculate the KPIs for the current repeat. """

        for kpi in self.sim_res.kpis:
            if kpi in ("EIOH_Cont", "Demand", "Sales"):
                # Per period
                self.sim_res.put_kpi(self.stats_sum[kpi][0],
                                     self.stats_sum[kpi][1],
                                     actual_nrperiods, kpi)
            elif (kpi in ("EOS", "ELT") and
                  self.stats_sum["EOL"][0] > dobr.EPS):
                # Per order line
                self.sim_res.put_kpi(self.stats_sum[kpi][0],
                    self.stats_sum[kpi][1],
                    self.stats_sum["EOL"][0], kpi)
            elif not kpi in ("Fillrate", "EST"):
                # Per review period
                self.sim_res.put_kpi(self.stats_sum[kpi][0],
                    self.stats_sum[kpi][1],
                    actual_nrperiods/self.sku_param["reviewperiod"], kpi)

        # Fill rate
        kpi = "Fillrate"
        if kpi in self.sim_res.kpis and self.stats_sum["Demand"][0] > dobr.EPS:
            self.sim_res.put_kpi(self.stats_sum["Sales"][0],
                self.stats_sum["Sales"][1],
                self.stats_sum["Demand"][0], kpi)
        # Sojourn time
        kpi = "EST"
        if kpi in self.sim_res.kpis and self.stats_sum["Sales"][0] > dobr.EPS:
            self.sim_res.put_kpi(self.stats_sum["EST"][0],
                self.stats_sum["EST"][1],
                self.stats_sum["Sales"][0], kpi)

        # Distributions (no weekpattern)
        if self.sim_res.report_pmfs:
            self.sim_res.pmf_scale(self.sku_param["reviewperiod"]/actual_nrperiods)


class SimInventory:
    """ Class main event loop. """

    def __init__(self, sku_list, sim_param, weekfractions=None):
        """ Initialize the simulation object. """

        # Simulation parameters
        self.sim_param = sim_param

        # Event heap
        self.fes = None

        # Create stock points for each SKU that requires results
        self.stocks = []
        for sku_param in sku_list:
            self.stocks.append(StockPoint(sku_param, sim_param, weekfractions))

    def single_repeat(self, repeat, output):
        """ Run a single repeat. """

        # Init event list
        self.fes = FES()
        # Start clock
        current_time = 0.
        weekday = 0

        for sp in self.stocks:
            # Reset result structure
            sp.sim_res.reset()
            # Reset inventory status and stats for the repeat
            sp.reset_repeat(repeat)
            if self.sim_param["weekpattern"]:
                # Add first demand for each weekday
                for weekday in range(MAXWEEKDAYS):
                    if sp.demand[weekday].mean > dobr.EPS:
                        iat, demand, demand_lifo = (
                            sp.demand[weekday].gen_next(0., first=True))
                        self.fes.add(EventData(weekday + iat,
                                               etype=DEMAND,
                                               item=sp,
                                               data=(demand, demand_lifo)))
            else:
                iat, demand, demand_lifo = sp.demand[weekday].gen_next(0.)
                self.fes.add(EventData(iat, etype=DEMAND, item=sp,
                                       data=(demand, demand_lifo)))

            # Add next inspection moment
            self.fes.add(EventData(current_time+1.,
                                   etype=INSPECT, item=sp))
            # Add next review moment
            self.fes.add(EventData(current_time+sp.sku_param["reviewperiod"],
                                   etype=REVIEW, item=sp))

        # Add next end of period moment
        self.fes.add(EventData(current_time+1., etype=ENDPERIOD))

        # Start the event loop
        while (self.fes.size() > 0
               and current_time < (self.sim_param["warmup"]
                                   + self.sim_param["nrperiods"])):

            # Get next event
            curr_event = self.fes.next()
            # Update the event timer and get the type of event and the sku
            current_time = curr_event.time
            curr_sp = curr_event.item

            # Action depends on current event type
            # print(self.current_time, "Event", curr_event.etype)
            if curr_event.etype == DEMAND:
                curr_sp.event_sales(current_time, curr_event.data)
                # Generate next demand
                iat, demand, demand_lifo = curr_sp.demand[weekday].gen_next(current_time)
                self.fes.add(EventData(current_time+iat,
                                       etype=DEMAND,
                                       item=curr_sp,
                                       data=(demand, demand_lifo)))
            elif curr_event.etype == INSPECT:
                curr_sp.event_inspect(current_time)
                # Add next inspection moment
                self.fes.add(EventData(current_time+1.,
                                       etype=INSPECT,
                                       item=curr_sp))
            elif curr_event.etype == REVIEW:
                delivery_batch = curr_sp.event_review(current_time,
                                                      weekday)
                # Add next potential delivery moment
                self.fes.add(EventData(delivery_batch.available,
                                       etype=DELIVERY,
                                       item=curr_sp,
                                       data=delivery_batch))
                # Add next review moment
                self.fes.add(EventData(current_time +
                                       curr_sp.sku_param["reviewperiod"],
                                       etype=REVIEW,
                                       item=curr_sp))
            elif curr_event.etype == DELIVERY:
                # Idle replenishment from the backroom?
                if curr_sp.sku_param["shelfspace"] > 0:
                    curr_sp.event_idle(current_time)
                # Receive the delivery
                curr_sp.event_delivery(current_time, curr_event.data)
            elif curr_event.etype == ENDPERIOD:
                # End of period for all SKUs
                for sp in self.stocks:
                    sp.event_endperiod(current_time)
                # Set weekday
                weekday = cycle_ref(weekday,
                                    weekpattern=self.sim_param["weekpattern"])
                # Add next end of period moment
                self.fes.add(EventData(current_time+1.,
                                       etype=ENDPERIOD))

        # Reporting for each sku
        output_list = []
        for sp in self.stocks:
            # Subtract remaining inventory from amount delivered
            sp.subtract_sumstat("ESUP", sp.inv.onhand)
            # Calculate KPI's
            # Number of periods corrected for actual simulation time
            sp.calc_repeatkpis(current_time-self.sim_param["warmup"])
            # Add simulation results to output list
            output_list.append(sp.sim_res)

        if output is None:
            # Single processing
            return output_list
        # Multi-processing: put the results in the output queue
        output.put(output_list)
