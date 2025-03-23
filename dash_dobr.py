# pylint: disable-msgXXX=C0302
# -*- coding: utf-8 -*-
""" This module generates the dashboard to evaluate periodic inventory
control systems.

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
import json
import tkinter as tk
from tkinter import ttk
from queue import Queue as qt
from threading import Thread
from multiprocessing import Process, Queue, cpu_count
from timeit import default_timer as timer
import numpy as np
import matplotlib
matplotlib.use("TkAgg")         # Required to show plots on the canvas
import matplotlib.pyplot as plt
import dobr_rsnq as dobr
import dobr_invsim

# Too many instance attributes
# pylint: disable-msg=R0902
# Too many branches
# pylint: disable-msg=R0912
# Too many arguments
# pylint: disable-msg=R0913
# Too many statements
# pylint: disable-msg=R0915

# Constants
BASE = 0                        # Base SKU
SIMCOL_HEADER = ["Simulation", "Precision", "Sim StDev"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
MAX_BARS = 100                  # Maximum number of horizontal bars in bar chart
MIN_REPEATS = 10                # Minimum number of repeats needed to determine precision
MIN_CORES = 4                   # Minimum number of required available cores
MAX_CORES = 10                  # Maximum number of cores that will be used
# Style constants for the dashboard
STR_EMPTYRESULT = " --.---"
STR_NOTYET = " ~~.~~~"
STR_ERROR = " ee.eee"
STYLE_EXACT = "black"
STYLE_APPROX = "blue"
STYLE_CI = "green"
STYLE_DIFF = "aquamarine"
BACK_COLOR = "#F0F0F0"

# Default dashboard is for the TU/e bachelor courses
DEFAULT_DASH = {
    "nr_skus" : 2,              # Number of different datasets or SKUs (maximum = 5)
    "analytic" : True,          # Report analytic results
    "target_rol": True,         # Report target reorder levels
    "retops": False,            # Report more analytic KPIs (perishability and capacity)
    "simulation": False,        # Report simulation results
    "sim_columns": 1,           # Number of simulation result columns (maximum = 3)
    "weekpattern": False}       # Include week pattern

def update_cycle(gui, gui_queue):
    """ Check if the simulation needs to run. """

    # Start the loop to check if simulation requests are made
    nr_repeats = 0
    while True:
        sim = gui_queue.get()
        # Initial request?
        if nr_repeats == 0:
            # Use multi processing?
            cores = cpu_count() - 1     # Available cpu cores minus 1 for dashboard
            if cores < MIN_CORES:
                cores = 1               # Use single core to avoid overhead
            elif cores > MAX_CORES:
                cores = MAX_CORES       # Limit number of cores to improve responsiveness

            # Start the timer
            if cores == 1:
                gui.update_statusbar("Simulation starting on a single CPU core")
            else:
                gui.update_statusbar(f"Simulation starting using {cores} CPU cores")
            start_job = timer()
            last_start = start_job

        # Execute the next repeat(s)
        if cores == 1:
            result_repeat = sim.single_repeat(nr_repeats+1, output=None)
            # Update the results
            gui.update_results_sim(result_repeat)
        else:
            # Create a queue instance for multi processing
            mp_q = Queue()
            # Create tasks for each core (one repeat for each core)
            tasks = []
            for task in range(cores):
                tasks.append(Process(target=sim.single_repeat,
                                     args=(nr_repeats+task+1, mp_q)))
            # Start the tasks
            for task in tasks:
                task.start()
            # Collect the results from the pipe in the queue
            # As soon as a task is ready, to reduce the storage in the pipe
            # since the pipe buffer can hold only 64Kb
            results_batch = [mp_q.get() for task in tasks]
            # Clear the process tasks when they are finished
            for task in tasks:
                task.join()
            # Read the results
            for result_repeat in results_batch:
                gui.update_results_sim(result_repeat)

        # Increase the number of repeats with the number of cores used
        nr_repeats += cores

        # Precision of target KPI
        msg, max_precision = update_sim_msg(gui, nr_repeats,
                                            sim.sim_param["targetkpi"])
        # Enough time elapsed to update the status bar?
        if timer() - last_start > 1.:
            # Write the result
            gui.update_statusbar(msg)
            # Reset status bar timer
            last_start = timer()
        # Sufficient repeats executed?
        if (nr_repeats < MIN_REPEATS or (nr_repeats < sim.sim_param["maxrepeats"]
          and max_precision > sim.sim_param["targetprecision"])):
            # Put next repeat(s) on the queue
            gui_queue.put(sim)
        else:
            msg = (f"Simulation finished after {nr_repeats} repeats"
                   + f" and {(timer()-start_job): 5.2f} seconds")
            gui.update_statusbar(msg)
            for sku in gui.skus:
                if sku.redo_sim:
                    if sim.sim_param["reportpmfs"]:
                        # Scale the PMFs
                        sku.pmfs_sim /= nr_repeats
                    sku.redo_sim = False
                    sku.redo_nbr = -1
                    gui.write_results_sim(sku)

            # Reset
            nr_repeats = 0

def update_sim_msg(gui, nr_repeats, target_kpi):
    """ Create a new status bar message and the max precision. """

    max_precision = -1.
    msg = f"Simulation running, with after {nr_repeats} repeats, KPI {target_kpi} ="
    for sku in gui.skus:
        if sku.redo_sim:
            msg += f" {sku.name} : {sku.mean_sim(target_kpi): 6.3f}"
            if nr_repeats >= MIN_REPEATS:
                msg += f" \u00B1 {sku.precision(target_kpi):6.4f}"
            if (max_precision < 0. or sku.precision(target_kpi) > max_precision):
                max_precision = sku.precision(target_kpi)
    return msg, max_precision

def validate_generic(entry_field,
                 noneg_val=False, nozero_val=False, pos_val=False,
                 int_val=False, ub_1=False, ub_1eps=False):
    """ Generic validator for DoBr dashboard entry fields."""
    # Initially, assume that the input is not correct
    correct = False
    # Retrieve the content from the emtry field
    content = entry_field.get()
    content = content.strip()   # Strip white spaces
    sep = content.find(":")     # Earlier error message (with :)?
    if sep!= -1:
        # Remove error text
        content = content[sep+1:].strip()
        # Replace field with new content string
        entry_field.delete(0, "end")
        entry_field.insert(0, content)
    # Is the input a number?
    try:
        # Convert the content to a float
        input_value = float(content)
        # Check the requested conditions and
        # add a error text if the validation fails
        if noneg_val and input_value < 0.:
            entry_field.insert(0, "Error (<0): ")
        elif nozero_val and abs(input_value) < dobr.EPS:
            entry_field.insert(0, "Error (=0): ")
        elif pos_val and input_value < dobr.EPS:
            entry_field.insert(0, "Error (<=0): ")
        elif int_val and int(input_value) != input_value:
            entry_field.insert(0, "Error (!=int): ")
        elif ub_1eps and input_value > (1.0-dobr.EPS):
            entry_field.insert(0, "Error (>=1): ")
        elif ub_1 and input_value > 1.0:
            entry_field.insert(0, "Error (>1): ")
        else:
            correct = True
        return correct, input_value
    except ValueError:
        # Add error text before input
        entry_field.insert(0, "Error (NaN): ")
        return False, 0

def display_precision(display_field, value):
    """Display the precision value on the dashboard."""
    display_field.configure(text=f"\u00B1 {value: 6.4f}",
                            fg=STYLE_EXACT, bg=BACK_COLOR)

def display_value(display_field, value, acc, int_val=False):
    """Display the value on the dashboard, together with the accuracy."""
    if dobr.not_error_code(value):
        if acc == dobr.VAL_EXACT:
            if int_val:
                display_field.configure(text=f"{value: 8}",
                                        fg=STYLE_EXACT, bg=BACK_COLOR)
            else:
                display_field.configure(text=f"{value: 12.3f}",
                                        fg=STYLE_EXACT, bg=BACK_COLOR)
        elif acc == dobr.VAL_APPROX:
            if int_val:
                display_field.configure(text=f"{value: 8}",
                                        fg=STYLE_APPROX, bg=BACK_COLOR)
            else:
                display_field.configure(text=f"{value: 12.3f}",
                                        fg=STYLE_APPROX, bg=BACK_COLOR)
        elif acc == dobr.VAL_CI:
            display_field.configure(text=f"{value: 12.3f}",
                                    fg=STYLE_CI, bg=BACK_COLOR)
        elif acc == dobr.VAL_NOTAPPLIC:
            display_field.configure(text="",
                                    fg=STYLE_EXACT, bg=BACK_COLOR)
        elif acc == dobr.VAL_NOTYET:
            display_field.configure(text=STR_NOTYET,
                                    fg=STYLE_EXACT, bg=BACK_COLOR)
        elif acc == dobr.VAL_EMPTY:
            display_field.configure(text=STR_EMPTYRESULT,
                                    fg=STYLE_EXACT, bg=BACK_COLOR)
    else:
        display_field.configure(text=STR_ERROR,
                                fg=STYLE_EXACT, bg=BACK_COLOR)

def pmf_range(apmf, x_min=0, eps=0.001):
    """ Determine the range with values greater than eps. """
    # Upper bound
    x_max = apmf.size - 1
    # Find smallest x with positive value
    while (apmf[x_min] < eps and x_min < x_max):
        x_min += 1
    # Find largest x with positive value
    while (apmf[x_max] < eps and x_max > x_min):
        x_max -= 1
    return [x_min, x_max]


class SkuData():
    """ Contains the SKU data. """
    def __init__(self, sku_nr, param_dict, kpis_list, report_pmfs=False):
        self.nbr = sku_nr
        if sku_nr == 0:
            self.name = "Base"
        elif sku_nr == 1:
            self.name = "Alternative"
        else:
            self.name = "Alternative "+str(sku_nr)
        self.param = param_dict
        # Initialize the parameter status
        self.correct = True
        self.not_yet = False
        # Initialize the analytic results
        self.redo_ana_system = True     # Renew the inventory system
        self.redo_ana_calc = True       # Renew the analytic calculations
        self.results_ana = {}
        for kpi in kpis_list:
            self.results_ana[kpi] = [0., dobr.VAL_NOTAPPLIC]
        # Initialize the simulation results
        self.redo_sim = True
        self.redo_nbr = -1
        self.nr_repeats = 0
        self.results_sim = {}
        for kpi in kpis_list:
            self.results_sim[kpi] = [0., 0., 0.]
        # Ininiatize the PMF data
        self.report_pmfs = report_pmfs
        self.pmfs_sim = None

    def reset_ana(self):
        """ Reset the analytic results. """
        self.redo_ana_calc = True
        for kpi in self.results_ana:
            # Value and accuracy
            self.results_ana[kpi] = [0., dobr.VAL_NOTAPPLIC]

    def value(self, kpi):
        """ Return the analytic KPI value. """
        return self.results_ana[kpi][0]

    def accuracy(self, kpi):
        """ Return the analytic KPI accuracy status. """
        return self.results_ana[kpi][1]

    def reset_sim(self, redo_nbr):
        """ Reset the simulation results. """
        self.redo_sim = True
        self.redo_nbr = redo_nbr
        self.nr_repeats = 0
        for kpi in self.results_sim:
            # Mean, Precision, and Sim StDev
            self.results_sim[kpi] = [0., 0., 0.]
        # Reset PMFs
        if self.report_pmfs:
            self.pmfs_sim = np.zeros((dobr_invsim.NR_PMFS, int(self.mioh())+1))

    def add_sim(self, result_sku_repeat):
        """ Add the results of the last simulation repeat. """
        self.nr_repeats += 1
        for kpi in result_sku_repeat.kpis:
            if kpi in self.results_sim:
                mean = result_sku_repeat.get_mean(kpi)
                var = result_sku_repeat.get_variance(kpi)
                self.results_sim[kpi][0] += mean
                self.results_sim[kpi][1] += mean**2
                self.results_sim[kpi][2] += var
        if self.report_pmfs:
            self.pmfs_sim += np.array(result_sku_repeat.pmfs)

    def mean_sim(self, kpi):
        """ Mean of the simulation results. """
        return self.results_sim[kpi][0]/self.nr_repeats

    def stdev_sim(self, kpi):
        """ Standard deviation of the simulation results. """
        var = 0.
        if self.results_sim[kpi][2] > dobr.EPS:
            var = self.results_sim[kpi][2]/self.nr_repeats
        return var**0.5

    def mioh(self):
        """ Maximum inventory on hand. """
        mioh = self.param["reorderlevel"] + max(self.param["ioq"],
                                                self.param["moq"])
        if self.param["distribution"] == "Discrete":
            mioh -= 1
        return mioh

    def precision(self, kpi):
        """ Precision of the simulation results. """
        return dobr_invsim.precision(self.results_sim[kpi][0],
                                     self.results_sim[kpi][1],
                                     self.nr_repeats)


class DobrDashboard():
    """ Contains the dashboard widgets. """
    def __init__(self, root, sim_queue=None):
        # Check for dashboard configuration file
        try:
            # Load the defaults from a json file
            with open("dash_config.json", encoding="utf8") as fp_def:
                self.dash = json.load(fp_def)
                # Correct settings
                # Maximum number of SKUs is 5
                self.dash["nr_skus"] = min(self.dash["nr_skus"], 5)
                if self.dash["simulation"]:
                    # No target reorder levels with simulation
                    self.dash["target_rol"] = False
        except OSError:
            # File with dashboard constants not found: use defaults
            self.dash = DEFAULT_DASH

        # Set the main window
        root.title(
            "DoBr dashboard for periodic review inventory systems"
            + f" (version {dobr.VERSION})")
        # Load TU/e logo
        root.iconbitmap("TUe-favicon-32px.ico")

        # Connect to the simulation queue
        self.sim_queue = sim_queue

        # The set of KPIs that will be displayed in this dashboard
        self.kpis_list = ["Fillrate", "Readyrate",
                           "EBO_L", "EBO_RL", "EIOH_L", "EIOH_RL",
                           "EOL", "EOS"]
        if self.dash["retops"] or self.dash["simulation"]:
            self.kpis_list.extend(["ESUP", "EST", "ENB", "EW",
                                   "POC", "EIBR", "ENIR", "ENCR",
                                   "EUA", "EUSL_L"])

        # Set the default base sku (0) and alternative skus
        try:
            # Load the defaults from a json file
            with open("dobr_defaults.json", encoding="utf8") as fp_def:
                base_data = json.load(fp_def)
        except OSError:
            # File with defaults not found
            base_data = dobr.DEFAULT_DATA
        # Correct if necessary, dependent on the dashboard configuration
        if self.dash["retops"] and not self.dash["simulation"]:
            # This dashboard can only deal with discrete demand distributions
            base_data["distribution"] = "Discrete"
        elif not self.dash["simulation"]:
            # This dashboard only deals with backordering and (R,s,nQ)
            base_data["lostsales"] = False
            base_data["moq"] = 0
            base_data["shelflife"] = 0
            base_data["shelfspace"] = 0

        # Simulation parameters
        try:
            with open("dobr_simdefaults.json", encoding="utf8") as fp_def:
                self.sim_param = dict(json.load(fp_def))
        except OSError:
            # File with parameters not found: use defaults
            self.sim_param = dict(dobr_invsim.DEFAULT_SIM)
        # Reset depending on dashboard cofiguration
        if not self.dash["weekpattern"]:
            self.sim_param["weekpattern"] = False

        # Validation functions of the simulation parameters
        self.sim_val_functions = {}
        self.new_simoptions = True

        # Create for each SKU an object in which we store
        # a shallow copy, using dict(), of the base SKU parameters
        self.skus = []
        for sku_nbr in range(self.dash["nr_skus"]):
            self.skus.append(SkuData(sku_nbr, dict(base_data), self.kpis_list,
                                     report_pmfs=self.sim_param["reportpmfs"]))

        if self.dash["nr_skus"] > 1:
            # Let the alternative skus have a different situation
            for sku_nbr in range(1, self.dash["nr_skus"]):
                self.skus[sku_nbr].param["reorderlevel"] = (
                    self.skus[0].param["reorderlevel"] + sku_nbr)

        if self.dash["weekpattern"]:
            # Default week pattern = Kahn & Schmittlein
            self.week_fractions = [0.12, 0.13, 0.13, 0.16, 0.18, 0.18, 0.1]

        # Validation functions of the parameters
        self.val_functions = {}

        # Default target values for fill rate and discrete ready rate
        if self.dash["target_rol"]:
            self.tfr = dobr.DEF_TARGET
            self.new_tfr = True
            self.trr = dobr.DEF_TARGET
            self.new_trr = True

        # Initialize inventory systems for each sku
        self.system_ana = [None] * self.dash["nr_skus"]

        # root_row_nr = 0
        root_col_nr = 0
        # Set the parameters frame
        self.frame_parameters = tk.LabelFrame(
            root, text="Parameters", padx=5, pady=5)
        self.frame_parameters.grid(row=0, column=0,
                                   padx=5, pady=5, sticky=tk.N)

        # Display the two column headers
        row_nr = 0
        if self.dash["nr_skus"] > 1:
            for sku in self.skus:
                tk.Label(self.frame_parameters, width=14,
                         text=sku.name).grid(row=row_nr, column=sku.nbr+1,
                                             sticky=tk.E)

        # Parameter distribution type (dist)
        row_nr += 1
        tk.Label(self.frame_parameters, text="Distribution").grid(
            row=row_nr, column=0, sticky=tk.W)
        self.dist_var = []
        self.opt_dist = []
        for sku in self.skus:
            self.dist_var.append(tk.StringVar())
            self.dist_var[sku.nbr].set(sku.param["distribution"])
            self.opt_dist.append(tk.OptionMenu(
                self.frame_parameters, self.dist_var[sku.nbr],
                *["Discrete", "Gamma", "Normal"],
                command=self._val_dist(sku)))
            self.opt_dist[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
            if self.dash["retops"] and not self.dash["simulation"]:
                self.opt_dist[sku.nbr].config(state="disabled", bg=BACK_COLOR)
            else:
                # Make the background white, like the entry widgets
                self.opt_dist[sku.nbr].configure(width=8, bg="white")

        # Parameter lost sales (ls)
        if self.dash["retops"] or self.dash["simulation"]:
            row_nr += 1
            tk.Label(self.frame_parameters, text="OOS situation").grid(
                row=row_nr, column=0, sticky=tk.W)
            self.chk_ls = []
            self.ls_var = []
            for sku in self.skus:
                self.ls_var.append(tk.IntVar())
                self.chk_ls.append(tk.Checkbutton(
                    self.frame_parameters, text="Lost sales", width=9,
                    justify=tk.RIGHT,
                    variable=self.ls_var[sku.nbr],
                    command=lambda sku=sku: self._val_ls(sku)))
                self.chk_ls[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
                self.chk_ls[sku.nbr].configure(bg="white", relief=tk.RIDGE)
                if sku.param["lostsales"]:
                    self.chk_ls[sku.nbr].toggle()

        # Parameter lead time
        row_nr += 1
        tk.Label(self.frame_parameters, text="Mean lead time").grid(
            row=row_nr, column=0, sticky=tk.W)
        self.ent_leadtime = []
        self.val_functions["leadtime"] = self._val_leadtime
        for sku in self.skus:
            # We use a lambda function to pass the sku_nr to the
            # validation command. By using sku=sku, we fix the parameter
            self.ent_leadtime.append(tk.Entry(
                self.frame_parameters, width=14, justify=tk.RIGHT,
                validate="focusout",
                validatecommand=lambda sku=sku: self._val_leadtime(sku)))
            self.ent_leadtime[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
            self.ent_leadtime[sku.nbr].insert(0, sku.param["leadtime"])

        # Parameter stdev_leadtime
        row_nr += 1
        tk.Label(self.frame_parameters, text="StDev lead time").grid(
            row=row_nr, column=0, sticky=tk.W)
        self.ent_stdev_leadtime = []
        self.val_functions["stdev_leadtime"] = self._val_stdev_leadtime
        for sku in self.skus:
            self.ent_stdev_leadtime.append(tk.Entry(
                self.frame_parameters, width=14, justify=tk.RIGHT,
                validate="focusout",
                validatecommand=lambda sku=sku: self._val_stdev_leadtime(sku)))
            self.ent_stdev_leadtime[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
            self.ent_stdev_leadtime[sku.nbr].insert(0, sku.param["stdev_leadtime"])

        # Parameter reviewperiod
        row_nr += 1
        tk.Label(self.frame_parameters, text="Review period").grid(
            row=row_nr, column=0, sticky=tk.W)
        self.ent_reviewperiod = []
        self.val_functions["reviewperiod"] = self._val_reviewperiod
        for sku in self.skus:
            self.ent_reviewperiod.append(tk.Entry(
                self.frame_parameters, width=14, justify=tk.RIGHT,
                validate="focusout",
                validatecommand=lambda sku=sku: self._val_reviewperiod(sku)))
            self.ent_reviewperiod[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
            self.ent_reviewperiod[sku.nbr].insert(0, sku.param["reviewperiod"])

        # Parameter mean period demand (mean)
        row_nr += 1
        tk.Label(self.frame_parameters, text="Mean period demand").grid(
            row=row_nr, column=0, sticky=tk.W)
        self.ent_mean = []
        self.val_functions["mean_perioddemand"] = self._val_mean
        for sku in self.skus:
            self.ent_mean.append(tk.Entry(
                self.frame_parameters, width=14, justify=tk.RIGHT,
                validate="focusout",
                validatecommand=lambda sku=sku: self._val_mean(sku)))
            self.ent_mean[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
            self.ent_mean[sku.nbr].insert(0, sku.param["mean_perioddemand"])

        # Parameter stdev period demand (stdev)
        row_nr += 1
        tk.Label(self.frame_parameters, text="StDev period demand").grid(
            row=row_nr, column=0, sticky=tk.W)
        self.ent_stdev = []
        self.val_functions["stdev_perioddemand"] = self._val_stdev
        for sku in self.skus:
            self.ent_stdev.append(tk.Entry(
                self.frame_parameters, width=14, justify=tk.RIGHT,
                validate="focusout",
                validatecommand=lambda sku=sku: self._val_stdev(sku)))
            self.ent_stdev[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
            self.ent_stdev[sku.nbr].insert(0, sku.param["stdev_perioddemand"])

        # Parameter ioq (case pack size)
        row_nr += 1
        tk.Label(self.frame_parameters, text="IOQ (case pack size)").grid(
            row=row_nr, column=0, sticky=tk.W)
        self.ent_ioq = []
        self.val_functions["ioq"] = self._val_ioq
        for sku in self.skus:
            self.ent_ioq.append(tk.Entry(
                self.frame_parameters, width=14, justify=tk.RIGHT,
                validate="focusout",
                validatecommand=lambda sku=sku: self._val_ioq(sku)))
            self.ent_ioq[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
            self.ent_ioq[sku.nbr].insert(0, sku.param["ioq"])

        # Parameter moq
        if self.dash["retops"] or self.dash["simulation"]:
            row_nr += 1
            tk.Label(self.frame_parameters, text="MOQ (0 -> equal to IOQ)").grid(
                row=row_nr, column=0, sticky=tk.W)
            self.ent_moq = []
            self.val_functions["moq"] = self._val_moq
            for sku in self.skus:
                self.ent_moq.append(tk.Entry(
                    self.frame_parameters, width=14, justify=tk.RIGHT,
                    validate="focusout",
                    validatecommand=lambda sku=sku: self._val_moq(sku)))
                self.ent_moq[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
                self.ent_moq[sku.nbr].insert(0, sku.param["moq"])

        # Parameter reorder level (rol)
        row_nr += 1
        tk.Label(self.frame_parameters, text="Reorder level").grid(
            row=row_nr, column=0, sticky=tk.W)
        self.ent_rol = []
        self.val_functions["reorderlevel"] = self._val_rol
        for sku in self.skus:
            self.ent_rol.append(tk.Entry(
                self.frame_parameters, width=14, justify=tk.RIGHT,
                validate="focusout",
                validatecommand=lambda sku=sku: self._val_rol(sku)))
            self.ent_rol[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
            self.ent_rol[sku.nbr].insert(0, sku.param["reorderlevel"])
            if sku.nbr > 0:
                # Initial, the reorder level is different for the alternative(s)
                self.ent_rol[sku.nbr].configure(bg="aquamarine")

        if self.dash["retops"] or self.dash["simulation"]:
            # Shelf life related parameters
            # Draw line
            row_nr += 1
            ttk.Separator(master=self.frame_parameters, orient=tk.HORIZONTAL).grid(
                row=row_nr, column=0, columnspan=self.dash["nr_skus"]+1,
                sticky=tk.W+tk.E)
            # Header
            # row_nr += 1
            # tk.Label(self.frame_parameters, text="Perishability parameters:").grid(
            #     row=row_nr, column=0, sticky=tk.W)

            # Parameter shelf life (sl)
            row_nr += 1
            tk.Label(self.frame_parameters, text="Shelf life (0 = \u221E)").grid(
                row=row_nr, column=0, sticky=tk.W)
            self.ent_shelflife = []
            self.val_functions["shelflife"] = self._val_shelflife
            for sku in self.skus:
                self.ent_shelflife.append(tk.Entry(
                    self.frame_parameters, width=14, justify=tk.RIGHT,
                    validate="focusout",
                    validatecommand=lambda sku=sku: self._val_shelflife(sku)))
                self.ent_shelflife[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
                self.ent_shelflife[sku.nbr].insert(0, sku.param["shelflife"])

            # Parameter fraction fifo
            row_nr += 1
            tk.Label(self.frame_parameters, text="Fraction FIFO withdrawal").grid(
                row=row_nr, column=0, sticky=tk.W)
            self.ent_fifo = []
            self.val_functions["fifo"] = self._val_fifo
            for sku in self.skus:
                self.ent_fifo.append(tk.Entry(
                    self.frame_parameters, width=14, justify=tk.RIGHT,
                    validate="focusout",
                    validatecommand=lambda sku=sku: self._val_fifo(sku)))
                self.ent_fifo[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
                self.ent_fifo[sku.nbr].insert(0, sku.param["fifo"])
                if not self.dash["simulation"]:
                    self.ent_fifo[sku.nbr].config(state="disabled")

            # Parameter EWA
            row_nr += 1
            tk.Label(self.frame_parameters, text="Modify IP").grid(
                row=row_nr, column=0, sticky=tk.W)
            self.chk_ewa = []
            self.ewa_var = []
            for sku in self.skus:
                self.ewa_var.append(tk.IntVar())
                self.chk_ewa.append(tk.Checkbutton(
                    self.frame_parameters, text="EWA", width=9, justify=tk.RIGHT,
                    variable=self.ewa_var[sku.nbr],
                    command=lambda sku=sku: self._val_ewa(sku)))
                self.chk_ewa[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
                self.chk_ewa[sku.nbr].configure(bg="white", relief=tk.RIDGE)
                if sku.param["EWA"]:
                    self.chk_ewa[sku.nbr].toggle()
                if not self.dash["simulation"]:
                    self.chk_ewa[sku.nbr].config(state="disabled", bg=BACK_COLOR)

            # Parameter ROS (restricted order size)
            row_nr += 1
            tk.Label(self.frame_parameters, text="Modify order").grid(
                row=row_nr, column=0, sticky=tk.W)
            self.chk_ros = []
            self.ros_var = []
            for sku in self.skus:
                self.ros_var.append(tk.IntVar())
                self.chk_ros.append(tk.Checkbutton(
                    self.frame_parameters, text="Restrict", width=9, justify=tk.RIGHT,
                    variable=self.ros_var[sku.nbr],
                    command=lambda sku=sku: self._val_ros(sku)))
                self.chk_ros[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
                self.chk_ros[sku.nbr].configure(bg="white", relief=tk.RIDGE)
                if sku.param["ROS"]:
                    self.chk_ros[sku.nbr].toggle()
                if not self.dash["simulation"]:
                    self.chk_ros[sku.nbr].config(state="disabled", bg=BACK_COLOR)

            # Capacity related parameters
            # Draw line
            row_nr += 1
            ttk.Separator(master=self.frame_parameters, orient=tk.HORIZONTAL).grid(
                row=row_nr, column=0, columnspan=self.dash["nr_skus"]+1,
                sticky=tk.W+tk.E)
            # Header
            # row_nr += 1
            # tk.Label(self.frame_parameters, text="Capacity parameters:").grid(
            #     row=row_nr, column=0, sticky=tk.W)

            # Parameter unit load capacity (unitcap)
            row_nr += 1
            tk.Label(self.frame_parameters, text="Unit load capacity").grid(
                row=row_nr, column=0, sticky=tk.W)
            self.ent_unitcap = []
            self.val_functions["unitcap"] = self._val_unitcap
            for sku in self.skus:
                self.ent_unitcap.append(tk.Entry(
                    self.frame_parameters, width=14, justify=tk.RIGHT,
                    validate="focusout",
                    validatecommand=lambda sku=sku: self._val_unitcap(sku)))
                self.ent_unitcap[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
                self.ent_unitcap[sku.nbr].insert(0, sku.param["unitcap"])

            # Parameter shelf space
            row_nr += 1
            tk.Label(self.frame_parameters, text="Shelf space (0 = \u221E)").grid(
                row=row_nr, column=0, sticky=tk.W)
            self.ent_shelfspace = []
            self.val_functions["shelfspace"] = self._val_shelfspace
            for sku in self.skus:
                self.ent_shelfspace.append(tk.Entry(
                    self.frame_parameters, width=14, justify=tk.RIGHT,
                    validate="focusout",
                    validatecommand=lambda sku=sku: self._val_shelfspace(sku)))
                self.ent_shelfspace[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
                self.ent_shelfspace[sku.nbr].insert(0, sku.param["shelfspace"])

            # Parameter concurrent (conc)
            row_nr += 1
            tk.Label(self.frame_parameters, text="Backroom replenishments").grid(
                row=row_nr, column=0, sticky=tk.W)
            self.chk_conc = []
            self.conc_var = []
            for sku in self.skus:
                self.conc_var.append(tk.IntVar())
                self.chk_conc.append(tk.Checkbutton(
                    self.frame_parameters, text="Concurrent", width=9, justify=tk.RIGHT,
                    variable=self.conc_var[sku.nbr],
                    command=lambda sku=sku: self._val_conc(sku)))
                self.chk_conc[sku.nbr].grid(row=row_nr, column=sku.nbr+1)
                self.chk_conc[sku.nbr].configure(bg="white", relief=tk.RIDGE)
                if sku.param["concurrent"]:
                    self.chk_conc[sku.nbr].toggle()
                if not self.dash["simulation"]:
                    self.chk_conc[sku.nbr].config(state="disabled", bg=BACK_COLOR)

        # Set the week pattern frame
        if self.dash["weekpattern"]:
            root_col_nr += 1
            self.frame_wp = tk.LabelFrame(
                root, text="Week pattern", padx=5, pady=5)
            self.frame_wp.grid(
                row=0, column=root_col_nr, padx=5, pady=5, sticky=tk.N)

            self.wp_var = tk.IntVar()
            self.chk_wp = tk.Checkbutton(
                self.frame_wp, text="Apply", width=6, justify=tk.RIGHT,
                variable=self.wp_var,
                command=self._val_wp)
            self.chk_wp.grid(row=0, column=0)
            self.chk_wp.configure(bg="white", relief=tk.RIDGE)
            if self.sim_param["weekpattern"]:
                self.chk_wp.toggle()

            tk.Label(self.frame_wp, text="Fraction").grid(
                row=0, column=1, sticky=tk.E)
            self.ent_wf = []
            for day in range(dobr_invsim.MAXWEEKDAYS):
                tk.Label(self.frame_wp, text=f"{WEEKDAYS[day]}").grid(
                    row=day+1, column=0, padx=5, sticky=tk.W)
                self.ent_wf.append(tk.Entry(
                    self.frame_wp, width=10, justify=tk.RIGHT,
                        validate="focusout",
                        validatecommand=lambda day=day: self._val_wf(day)))
                self.ent_wf[day].grid(row=day+1, column=1, padx=5)
                self.ent_wf[day].insert(0, self.week_fractions[day])

        # Set the KPI frame
        root_col_nr += 1
        self.frame_kpis = tk.LabelFrame(
            root, text="Key Performance Indicators (KPIs)",
            padx=5, pady=5)
        self.frame_kpis.grid(row=0, column=root_col_nr, padx=5, pady=5)

        # Determine the number of result columns per SKU
        self.sku_columns = 0
        if self.dash["analytic"]:
            self.sku_columns += 1
        if self.dash["simulation"]:
            self.sku_columns += self.dash["sim_columns"]

        # Display the column headers
        row_nr = 0
        if self.dash["nr_skus"] > 1:
            for sku_nbr in range(self.dash["nr_skus"]):
                col_nr = sku_nbr * self.sku_columns
                tk.Label(self.frame_kpis, width=14,
                         text=self.skus[sku_nbr].name).grid(
                             row=row_nr, column=1+col_nr, sticky=tk.E)
        # We need an additional header row for the different results per SKU
        if self.dash["simulation"]:
            row_nr += 1
            for sku_nbr in range(self.dash["nr_skus"]):
                col_nr = sku_nbr * self.sku_columns
                if self.dash["analytic"]:
                    tk.Label(self.frame_kpis, text="Analytic").grid(
                        row=row_nr, column=1+col_nr, sticky=tk.W)
                    col_nr += 1
                for sim_col in range(self.dash["sim_columns"]):
                    tk.Label(self.frame_kpis,
                             text=SIMCOL_HEADER[sim_col]).grid(
                                 row=row_nr, column=1+col_nr+sim_col,
                                 sticky=tk.W)

        # Create labels for each KPI
        self.lbl_results = {}
        for kpi in self.kpis_list:
            # Separator line?
            if kpi in ("ESUP", "POC", "EUA"):
                # Draw line
                row_nr += 1
                ttk.Separator(master=self.frame_kpis, orient=tk.HORIZONTAL).grid(
                    row=row_nr, column=0,
                    columnspan=1+self.dash["nr_skus"]*self.sku_columns,
                    sticky=tk.W+tk.E)
            row_nr += 1
            tk.Label(self.frame_kpis, text=dobr.KPI_CATALOGUE[kpi]).grid(
                row=row_nr, column=0, sticky=tk.W)
            # Add list to the dictionary entry
            self.lbl_results[kpi] = [None] * self.dash["nr_skus"] * self.sku_columns
            for sku in self.skus:
                for sku_col_nr in range(self.sku_columns):
                    col_nr = sku.nbr*self.sku_columns + sku_col_nr
                    if dobr.applicable_kpi(kpi, sku.param):
                        acc = dobr.VAL_EMPTY
                    else:
                        acc = dobr.VAL_NOTAPPLIC
                    self.lbl_results[kpi][col_nr] = tk.Label(
                        self.frame_kpis, width=14, text=STR_EMPTYRESULT)
                    self.lbl_results[kpi][col_nr].grid(
                        row=row_nr, column=col_nr+1, sticky=tk.E)
                    display_value(self.lbl_results[kpi][col_nr], 0., acc)

        # Set the recalculation frame without simulation and week pattern
        if not self.dash["simulation"] and not self.dash["weekpattern"]:
            self.frame_calc = tk.LabelFrame(
                root, text="(Re)calculation", padx=5, pady=5)
            self.frame_calc.grid(
                row=1, column=0, padx=5, pady=5, sticky=tk.N)

            row_nr = 0
            # Set the recalc button
            self.btn_recalc_kpis = tk.Button(
                self.frame_calc, width=14, text="Recalculate KPIs",
                command=self._recalc_kpis)
            self.btn_recalc_kpis.grid(row=row_nr, column=0,
                                      padx=5, pady=3, sticky=tk.W+tk.E)
            # Print output
            self.btn_print_ana = tk.Button(
                self.frame_calc, width=14, text="Output to console",
                command=self._print_kpis_ana_all)
            self.btn_print_ana.grid(row=row_nr, column=1,
                                    padx=5, pady=2)
            self.btn_print_ana["state"] = tk.DISABLED

            # Plots
            self.btn_plot_ioh_l = tk.Button(
                self.frame_calc, width=14, text="IOH(L) graph",
                command=self._plot_ioh_l)
            self.btn_plot_ioh_l.grid(row=row_nr+1, column=0)
            self.btn_plot_ioh_l["state"] = tk.DISABLED

            self.btn_plot_ioh_rl = tk.Button(
                self.frame_calc, width=14, text="IOH(R+L) graph",
                command=self._plot_ioh_rl)
            self.btn_plot_ioh_rl.grid(row=row_nr+1, column=1)
            self.btn_plot_ioh_rl["state"] = tk.DISABLED

            if self.dash["target_rol"]:
                # Set the ROL frame
                self.frame_rol = tk.LabelFrame(
                    root, text="Target reorder levels (ROLs)", padx=5, pady=5)
                self.frame_rol.grid(row=1, column=1, padx=5, pady=5)

                # Display the column headers
                row_nr = 0
                tk.Label(self.frame_rol, text="Target").grid(
                    row=row_nr, column=1, sticky=tk.E)
                if self.dash["nr_skus"] == 1:
                    tk.Label(self.frame_rol, width=14, text="ROL").grid(
                        row=row_nr, column=2, sticky=tk.E)
                else:
                    for sku_nbr in range(self.dash["nr_skus"]):
                        tk.Label(self.frame_rol, width=14,
                                 text=self.skus[sku_nbr].name).grid(
                                     row=row_nr, column=2+sku_nbr, sticky=tk.E)

                # Target fill rate reorder level (tfr)
                row_nr += 1
                tk.Label(self.frame_rol, text="Fill rate \u2265").grid(
                    row=row_nr, column=0, sticky=tk.W)
                self.ent_tfr = tk.Entry(
                    self.frame_rol, width=12, justify=tk.RIGHT, validate="focusout",
                    validatecommand=self._val_tfr)
                self.ent_tfr.grid(row=row_nr, column=1)
                self.ent_tfr.insert(0, self.tfr)
                self.rol_result_fr = []
                for sku_nbr in range(self.dash["nr_skus"]):
                    self.rol_result_fr.append(tk.Label(
                        self.frame_rol, width=14,
                        text=STR_EMPTYRESULT))
                    self.rol_result_fr[sku_nbr].grid(row=row_nr, column=sku_nbr+2,
                                                     sticky=tk.E)

                # Target ready rate reorder level (trr)
                row_nr += 1
                tk.Label(self.frame_rol, text="Discrete ready rate \u2265    ").grid(
                    row=row_nr, column=0, sticky=tk.W)
                self.ent_trr = tk.Entry(
                    self.frame_rol, width=12, justify=tk.RIGHT, validate="focusout",
                    validatecommand=self._val_trr)
                self.ent_trr.grid(row=row_nr, column=1)
                self.ent_trr.insert(0, self.trr)
                self.rol_result_rr = []
                for sku_nbr in range(self.dash["nr_skus"]):
                    self.rol_result_rr.append(tk.Label(
                        self.frame_rol, width=14,
                        text=STR_EMPTYRESULT))
                    self.rol_result_rr[sku_nbr].grid(row=row_nr, column=sku_nbr+2,
                                                     sticky=tk.E)

        else:
            # Set the simulation options frame
            self.frame_sopt = tk.LabelFrame(
                root, text="Simulation options", padx=9, pady=12)
            self.frame_sopt.grid(
                row=1, column=0, padx=5, pady=5, sticky=tk.W)
            row_nr = 0

            # Parameter: poisson process (pp)
            # tk.Label(self.frame_sopt, text="Arrival process").grid(
            #     row=row_nr, column=0, sticky=tk.W)
            # self.sim_val_functions["poissonprocess"] = self._val_pp
            # self.pp_var = tk.IntVar()
            # self.chk_pp = tk.Checkbutton(
            #     self.frame_sopt, text="Poisson", width=9, justify=tk.RIGHT,
            #     variable=self.pp_var,
            #     command=self._val_pp)
            # self.chk_pp.grid(row=row_nr, column=1)
            # self.chk_pp.configure(bg="white", relief=tk.RIDGE)
            # if self.sim_param["poissonprocess"]:
            #     self.chk_pp.toggle()
            # # TEMP
            # self.chk_pp["state"] = tk.DISABLED

            # Parameter: maximum number of subruns (maxr)
            row_nr += 1
            tk.Label(self.frame_sopt, text="Max number of subruns").grid(
                row=row_nr, column=0, sticky=tk.W)
            self.sim_val_functions["maxrepeats"] = self._val_maxr
            self.ent_maxr = tk.Entry(
                self.frame_sopt, width=14, justify=tk.RIGHT,
                validate="focusout", validatecommand=self._val_maxr)
            self.ent_maxr.grid(row=row_nr, column=1)
            self.ent_maxr.insert(0, self.sim_param["maxrepeats"])

            # Parameter: warmup periods per subruns
            row_nr += 1
            tk.Label(self.frame_sopt, text="Subrun warmup [periods]").grid(
                row=row_nr, column=0, sticky=tk.W)
            self.sim_val_functions["warmup"] = self._val_warmup
            self.ent_warmup = tk.Entry(
                self.frame_sopt, width=14, justify=tk.RIGHT,
                validate="focusout", validatecommand=self._val_warmup)
            self.ent_warmup.grid(row=row_nr, column=1)
            self.ent_warmup.insert(0, self.sim_param["warmup"])

            # Parameter: subrun periods (nrp)
            row_nr += 1
            tk.Label(self.frame_sopt, text="Subrun length [periods]").grid(
                row=row_nr, column=0, sticky=tk.W)
            self.sim_val_functions["nrperiods"] = self._val_nrp
            self.ent_nrp = tk.Entry(
                self.frame_sopt, width=14, justify=tk.RIGHT,
                validate="focusout", validatecommand=self._val_nrp)
            self.ent_nrp.grid(row=row_nr, column=1)
            self.ent_nrp.insert(0, self.sim_param["nrperiods"])

            # Set the recalculation frame with simulation
            self.frame_calc = tk.LabelFrame(
                root, text="(Re)calculation", padx=5, pady=5)
            self.frame_calc.grid(
                row=1, column=root_col_nr, padx=5, pady=5, sticky=tk.N)

            col_nr = 0
            if self.dash["analytic"]:
                # Button: recalculate KPIs
                self.btn_recalc_kpis = tk.Button(
                    self.frame_calc, text="Recalculate KPIs",
                    command=self._recalc_kpis)
                self.btn_recalc_kpis.grid(
                    row=0, column=col_nr, padx=6, pady=0, sticky=tk.W+tk.E)

                # Button: print analytic KPIs
                self.btn_print_ana = tk.Button(
                    self.frame_calc, text="Print analytic KPIs",
                    command=self._print_kpis_ana_all)
                self.btn_print_ana.grid(
                    row=1, column=col_nr, padx=6, pady=0, sticky=tk.W+tk.E)
                self.btn_print_ana["state"] = tk.DISABLED
                col_nr += 1

            # Button: simulate KPIs
            self.btn_sim_kpis = tk.Button(
                self.frame_calc, text="Simulate KPIs",
                command=self._sim_kpis)
            self.btn_sim_kpis.grid(
                row=0, column=col_nr, padx=6, pady=0, sticky=tk.W+tk.E)

            # # Button: print simulated KPIs
            self.btn_print_sim = tk.Button(
                self.frame_calc, text="Print simulated KPIs",
                command=self._print_kpis_sim_all)
            self.btn_print_sim.grid(
                row=1, column=col_nr, padx=6, pady=0, sticky=tk.W+tk.E)
            self.btn_print_sim["state"] = tk.DISABLED

            # # Parameter: simulate PMFs
            # self.sim_val_functions["reportpmfs"] = self._val_sg
            # self.sg_var = tk.IntVar()
            # self.chk_sg = tk.Checkbutton(
            #     self.frame_calc, text="Simulate graphs (no multiprocessing)",
            #     justify=tk.LEFT,
            #     variable=self.sg_var,
            #     command=self._val_sg)
            # self.chk_sg.grid(row=2, column=col_nr, columnspan=2)
            # # self.chk_sg.configure(bg="white", relief=tk.RIDGE)
            # if self.sim_param["reportpmfs"]:
            #     self.chk_sg.toggle()

            col_nr += 1
            # Button: display IOH(L) graph
            self.btn_plot_ioh_l = tk.Button(
                self.frame_calc, width=14, text="IOH(L) graph", command=self._plot_ioh_l)
            self.btn_plot_ioh_l.grid(row=0, column=col_nr, padx=5, pady=0)
            self.btn_plot_ioh_l["state"] = tk.DISABLED

            # Button: display IOH(R+L) graph
            self.btn_plot_ioh_rl = tk.Button(
                self.frame_calc, width=14, text="IOH(R+L) graph", command=self._plot_ioh_rl)
            self.btn_plot_ioh_rl.grid(row=1, column=col_nr, padx=5, pady=0)
            self.btn_plot_ioh_rl["state"] = tk.DISABLED

            col_nr += 1
            # Button: display IP graph
            self.btn_plot_ip = tk.Button(
                self.frame_calc, width=14, text="IP+ graph", command=self._plot_ip_plus)
            self.btn_plot_ip.grid(row=0, column=col_nr, padx=5, pady=0)
            self.btn_plot_ip["state"] = tk.DISABLED

            # Button: display OS graph
            self.btn_plot_os = tk.Button(
                self.frame_calc, width=14, text="OS graph", command=self._plot_os)
            self.btn_plot_os.grid(row=1, column=col_nr, padx=5, pady=0)
            self.btn_plot_os["state"] = tk.DISABLED

            # # Button: display TBO graph
            # self.btn_plot_tbo = tk.Button(
            #     self.frame_calc, width=10, text="TBO graph", command=self.plot_tbo)
            # self.btn_plot_tbo.grid(row=1, column=col_nr, padx=5, pady=0)
            # self.btn_plot_tbo["state"] = tk.DISABLED

            # col_nr += 1
            # # Button: display W graph
            # self.btn_plot_w = tk.Button(
            #     self.frame_calc, width=10, text="W graph", command=self.plot_w)
            # self.btn_plot_w.grid(row=0, column=col_nr, padx=5, pady=0)
            # self.btn_plot_w["state"] = tk.DISABLED

        # Set the status bar
        self.statusbar = tk.Label(
            root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.grid(row=2, column=0, columnspan=root_col_nr+1,
                            sticky=tk.W+tk.E)

    def _reset_kpi_results(self, sku=None, sim_only=False, target_rol=True):
        """ Resets the buttons and the result columns.

        Keyword arguments:
            sku        -- Which SKU results need to be reset (None means all)
            sim_only   -- Only the simulation results need to be reset
            target_rol -- Only the target reorder level (rol) results need to be reset
        """

        # Set the recalculation buttons to NORMAL (=active)
        # and disable the print and plot buttons
        if self.dash["analytic"] and (not sim_only):
            self.btn_recalc_kpis["state"] = tk.NORMAL
            self.btn_print_ana["state"] = tk.DISABLED
        self.btn_plot_ioh_l["state"] = tk.DISABLED
        self.btn_plot_ioh_rl["state"] = tk.DISABLED
        # The other plots depend only on simulation
        if self.dash["simulation"]:
            self.btn_sim_kpis["state"] = tk.NORMAL
            self.btn_print_sim["state"] = tk.DISABLED
            self.btn_plot_ip["state"] = tk.DISABLED
            self.btn_plot_os["state"] = tk.DISABLED

            # # For TBO and waste, we have no analytical results yet
            # self.btn_plot_tbo["state"] = tk.DISABLED
            # self.btn_plot_w["state"] = tk.DISABLED

        if self.dash["analytic"] and not sim_only:
            for sku_obj in self.skus:
                if (sku is None) or sku_obj == sku:
                    self._reset_ana_results(sku_obj)
        if self.dash["simulation"]:
            for sku_obj in self.skus:
                if (sku is None) or sku_obj == sku:
                    self._reset_sim_results(sku_obj)

        # Reset the target rol results
        if self.dash["target_rol"] and target_rol:
            self._reset_fr_results(sku=sku)
            self._reset_rr_results(sku=sku)

        # Empty the status bar
        self.update_statusbar("")

    def _reset_ana_results(self, sku):
        # Reset the previous values
        sku.reset_ana()
        # Reset the analytic result columns
        col_nr = sku.nbr*self.sku_columns
        # Reset the analytics column
        for kpi in sku.results_ana:
            value = 0.
            if not sku.correct:
                acc= dobr.VAL_ERROR
                value = -9999
            else:
                if dobr.applicable_kpi(kpi, sku.param):
                    if sku.not_yet:
                        acc = dobr.VAL_NOTYET
                    else:
                        acc = dobr.VAL_EMPTY
                else:
                    acc = dobr.VAL_NOTAPPLIC
            display_value(self.lbl_results[kpi][col_nr], value, acc)

    def _reset_sim_results(self, sku):
        # Reset the previous values
        sku.reset_sim(sku.nbr)
        # Reset the simulation result columns
        col_nr = sku.nbr*self.sku_columns
        if self.dash["analytic"]:
            col_nr += 1
        for sku_col_nr in range(self.dash["sim_columns"]):
            for kpi in sku.results_sim:
                if dobr.applicable_kpi(kpi, sku.param):
                    acc = dobr.VAL_EMPTY
                else:
                    acc = dobr.VAL_NOTAPPLIC
                display_value(self.lbl_results[kpi][col_nr+sku_col_nr], 0., acc)

    def _reset_fr_results(self, sku=None):
        """ Resets the results columns in the ROL frame. """
        self.btn_recalc_kpis["state"] = tk.NORMAL
        if sku is None:
            # Reset all result columns
            for sku_obj in self.skus:
                if sku_obj.not_yet:
                    self.rol_result_fr[sku_obj.nbr].configure(
                        text=STR_NOTYET,
                        fg=STYLE_EXACT, bg=BACK_COLOR)
                else:
                    self.rol_result_fr[sku_obj.nbr].configure(
                        text=STR_EMPTYRESULT,
                        fg=STYLE_EXACT, bg=BACK_COLOR)
        else:
            if sku.not_yet:
                self.rol_result_fr[sku.nbr].configure(
                    text=STR_NOTYET,
                    fg=STYLE_EXACT, bg=BACK_COLOR)
            else:
                self.rol_result_fr[sku.nbr].configure(
                    text=STR_EMPTYRESULT,
                    fg=STYLE_EXACT, bg=BACK_COLOR)
        # Empty the status bar
        self.update_statusbar("")

    def _reset_rr_results(self, sku=None):
        """ Resets the results columns in the ROL frame. """
        self.btn_recalc_kpis["state"] = tk.NORMAL
        if sku is None:
            # Reset all result columns
            for sku_obj in self.skus:
                if sku_obj.not_yet:
                    self.rol_result_fr[sku_obj.nbr].configure(
                        text=STR_NOTYET,
                        fg=STYLE_EXACT, bg=BACK_COLOR)
                else:
                    self.rol_result_rr[sku_obj.nbr].configure(
                        text=STR_EMPTYRESULT,
                        fg=STYLE_EXACT, bg=BACK_COLOR)
        else:
            if sku.not_yet:
                self.rol_result_fr[sku.nbr].configure(
                    text=STR_NOTYET,
                    fg=STYLE_EXACT, bg=BACK_COLOR)
            else:
                self.rol_result_rr[sku.nbr].configure(
                    text=STR_EMPTYRESULT,
                    fg=STYLE_EXACT, bg=BACK_COLOR)
        # Empty the status bar
        self.update_statusbar("")

    def highlight_diff(self, widget_list, data_key, sku_nbr):
        """ Highlights the field in the alternative parameter column
        if its value differs from the field in the base parameter column.
        """
        if self.dash["nr_skus"] > 1:
            if sku_nbr == BASE:
                sku_nbr = 1
            diff = False
            if isinstance(widget_list[BASE], tk.Entry):
                if widget_list[BASE].get() != widget_list[sku_nbr].get():
                    diff = True
            else:
                if self.skus[BASE].param[data_key] != self.skus[sku_nbr].param[data_key]:
                    diff = True
            if diff:
                widget_list[sku_nbr].configure(bg="aquamarine")
            else:
                widget_list[sku_nbr].configure(bg="white")

    def _val_dist(self, sku):
        """ Wrapper for the validator."""
        def _wrapped_val_dist(new):
            """ Validate the distribution parameter. """
            sku.redo_ana_system = True
            sku.redo_ana_calc = True
            sku.redo_sim = True
            sku.param["distribution"] = new
            self.highlight_diff(self.opt_dist, "distribution", sku.nbr)
            self._reset_kpi_results(sku)
            return True
        return _wrapped_val_dist

    def _val_ls(self, sku):
        """ Validate the lost sales parameter. """
        self._reset_kpi_results(sku=sku)
        sku.redo_ana_system = True
        sku.redo_ana_calc = True
        sku.redo_sim = True
        sku.param["lostsales"] = bool(self.ls_var[sku.nbr].get())
        self.highlight_diff(self.chk_ls, "lostsales", sku.nbr)
        return True

    def _val_leadtime(self, sku):
        """ Validate the lead time parameter. """
        self.highlight_diff(self.ent_leadtime, "leadtime", sku.nbr)
        correct, input_value = validate_generic(self.ent_leadtime[sku.nbr],
                                     noneg_val=True)
        if correct and sku.param["leadtime"] != input_value:
            sku.param["leadtime"] = input_value
            self._reset_kpi_results(sku=sku)
            sku.redo_ana_system = True
            sku.redo_ana_calc = True
            sku.redo_sim = True
        return correct

    def _val_stdev_leadtime(self, sku):
        """ Validate the stdev lead time parameter. """
        self.highlight_diff(self.ent_stdev_leadtime, "stdev_leadtime", sku.nbr)
        correct, input_value = validate_generic(self.ent_stdev_leadtime[sku.nbr],
                                     noneg_val=True)
        if correct and sku.param["stdev_leadtime"] != input_value:
            sku.param["stdev_leadtime"] = input_value
            self._reset_kpi_results(sku=sku)
            sku.redo_ana_system = True
            sku.redo_ana_calc = True
            sku.redo_sim = True
        return correct

    def _val_reviewperiod(self, sku):
        """ Validate the review period parameter. """
        self.highlight_diff(self.ent_reviewperiod, "reviewperiod", sku.nbr)
        correct, input_value = validate_generic(self.ent_reviewperiod[sku.nbr],
                                     pos_val=True)
        if correct and sku.param["reviewperiod"] != input_value:
            sku.param["reviewperiod"] = input_value
            self._reset_kpi_results(sku=sku)
            sku.redo_ana_system = True
            sku.redo_ana_calc = True
            sku.redo_sim = True
        return correct

    def _val_mean(self, sku):
        """ Validate the mean period demand parameter. """
        self.highlight_diff(self.ent_mean, "mean_perioddemand", sku.nbr)
        correct, input_value = validate_generic(self.ent_mean[sku.nbr],
                                     pos_val=True)
        if correct and sku.param["mean_perioddemand"] != input_value:
            sku.param["mean_perioddemand"] = input_value
            self._reset_kpi_results(sku=sku)
            sku.redo_ana_system = True
            sku.redo_ana_calc = True
            sku.redo_sim = True
        return correct

    def _val_stdev(self, sku):
        """ Validate the stdev period demand parameter. """
        self.highlight_diff(self.ent_stdev, "stdev_perioddemand", sku.nbr)
        correct, input_value = validate_generic(self.ent_stdev[sku.nbr],
                                     nozero_val=True)
        if correct and input_value < -dobr.EPS:
            # Apply the power law
            mean = sku.param["mean_perioddemand"]
            input_value = round(dobr.stdev_powerlaw(mean), 3)
            self.ent_stdev[sku.nbr].delete(0, tk.END)
            self.ent_stdev[sku.nbr].insert(0, input_value)
        if correct and sku.param["stdev_perioddemand"] != input_value:
            sku.param["stdev_perioddemand"] = input_value
            self._reset_kpi_results(sku=sku)
            sku.redo_ana_system = True
            sku.redo_ana_calc = True
            sku.redo_sim = True
        return correct

    def _val_ioq(self, sku):
        """ Validate the ioq parameter. """
        self.highlight_diff(self.ent_ioq, "ioq", sku.nbr)
        if sku.param["distribution"] == "Discrete":
            correct, input_value = validate_generic(self.ent_ioq[sku.nbr],
                                     noneg_val=True,
                                     nozero_val=True, int_val=True)
        else:
            correct, input_value = validate_generic(self.ent_ioq[sku.nbr],
                                     noneg_val=True)
        if correct and sku.param["ioq"] != input_value:
            if sku.param["distribution"] == "Discrete":
                sku.param["ioq"] = int(input_value)
            else:
                sku.param["ioq"] = input_value
            self._reset_kpi_results(sku=sku)
            sku.redo_ana_system = True
            sku.redo_ana_calc = True
            sku.redo_sim = True
        return correct

    def _val_moq(self, sku):
        """ Validate the moq parameter. """
        self.highlight_diff(self.ent_moq, "moq", sku.nbr)
        if sku.param["distribution"] == "Discrete":
            correct, input_value = validate_generic(self.ent_moq[sku.nbr],
                                     noneg_val=True, int_val=True)
            if sku.param["ioq"] > 0:
                ioq_mult = input_value/sku.param["ioq"]
            else:
                ioq_mult = 1
            if correct and abs(ioq_mult -int(ioq_mult)) > dobr.EPS:
                correct = False
                self.ent_moq[sku.nbr].insert(0, "Error (!=n*ioq): ")
        else:
            correct, input_value = validate_generic(self.ent_moq[sku.nbr],
                                     noneg_val=True)
            if correct and input_value > dobr.EPS:
                # No MOQ allowed for continuous distributions
                correct = False
                self.ent_moq[sku.nbr].insert(0, "Error (!= 0): ")
        if correct and sku.param["moq"] != input_value:
            if sku.param["distribution"] == "Discrete":
                sku.param["moq"] = int(input_value)
            else:
                sku.param["moq"] = input_value
            self._reset_kpi_results(sku=sku)
            sku.redo_ana_system = True
            sku.redo_ana_calc = True
            sku.redo_sim = True
        return correct

    def _val_rol(self, sku):
        """ Validate the reorder level parameter. """
        self.highlight_diff(self.ent_rol, "reorderlevel", sku.nbr)
        if sku.param["distribution"] == "Discrete":
            correct, input_value = validate_generic(
                self.ent_rol[sku.nbr], int_val=True)
        else:
            correct, input_value = validate_generic(
                self.ent_rol[sku.nbr])
        if correct and sku.param["reorderlevel"] != input_value:
            if sku.param["distribution"] == "Discrete":
                sku.param["reorderlevel"] = int(input_value)
            else:
                sku.param["reorderlevel"] = input_value
            self._reset_kpi_results(sku=sku, target_rol=False)
            sku.redo_ana_calc = True
            sku.redo_sim = True
        return correct

    def _val_shelflife(self, sku):
        """ Validate the shelf life parameter. """
        self.highlight_diff(self.ent_shelflife, "shelflife", sku.nbr)
        correct, input_value = validate_generic(self.ent_shelflife[sku.nbr],
                                     noneg_val=True, int_val=True)
        if correct and (not sku.param["lostsales"] and input_value > 0):
            correct = False
            self.ent_shelflife[sku.nbr].insert(0, "Error (!=LS): ")
        if correct and 0 < input_value <= sku.param["reviewperiod"]:
            correct = False
            self.ent_shelflife[sku.nbr].insert(0, "Error (<=R): ")
        if correct and sku.param["shelflife"] != input_value:
            sku.param["shelflife"] = int(input_value)
            self._reset_kpi_results(sku=sku)
            sku.redo_ana_system = True
            sku.redo_ana_calc = True
            sku.redo_sim = True
        return correct

    def _val_fifo(self, sku):
        """ Validate the fifo parameter. """
        self.highlight_diff(self.ent_fifo, "fifo", sku.nbr)
        correct, input_value = validate_generic(self.ent_fifo[sku.nbr],
                                     noneg_val=True, ub_1=True)
        if correct and sku.param["fifo"] != input_value:
            sku.param["fifo"] = input_value
            self._reset_kpi_results(sku=sku)
            sku.redo_ana_calc = True
            sku.redo_sim = True
        return correct

    def _val_ewa(self, sku):
        """ Validate the EWA parameter. """
        self._reset_kpi_results(sku=sku)
        sku.redo_ana_calc = True
        sku.redo_sim = True
        sku.param["EWA"] = bool(self.ewa_var[sku.nbr].get())
        self.highlight_diff(self.chk_ewa, "EWA", sku.nbr)
        return True

    def _val_ros(self, sku):
        """ Validate the ROS parameter. """
        self._reset_kpi_results(sku=sku)
        sku.redo_ana_calc = True
        sku.redo_sim = True
        sku.param["ROS"] = bool(self.ros_var[sku.nbr].get())
        self.highlight_diff(self.chk_ros, "ROS", sku.nbr)
        return True

    def _val_unitcap(self, sku):
        """ Validate the unit load capacity parameter. """
        self.highlight_diff(self.ent_unitcap, "unitcap", sku.nbr)
        if sku.param["distribution"] == "Discrete":
            correct, input_value = validate_generic(self.ent_unitcap[sku.nbr],
                                     noneg_val=True, int_val=True)
            if sku.param["ioq"] > 0:
                ioq_mult = input_value/sku.param["ioq"]
            else:
                ioq_mult = 1
            if correct and abs(ioq_mult -int(ioq_mult)) > dobr.EPS:
                correct = False
                self.ent_unitcap[sku.nbr].insert(0, "Error (!=n*ioq): ")
        else:
            correct, input_value = validate_generic(self.ent_unitcap[sku.nbr],
                                     noneg_val=True)
        if correct and sku.param["unitcap"] != input_value:
            if sku.param["distribution"] == "Discrete":
                sku.param["unitcap"] = int(input_value)
            else:
                sku.param["unitcap"] = input_value
            self._reset_kpi_results(sku=sku, target_rol=False)
            sku.redo_ana_calc = True
            sku.redo_sim = True
        return correct

    def _val_shelfspace(self, sku):
        """ Validate the shelf space parameter. """
        self.highlight_diff(self.ent_shelfspace, "shelfspace", sku.nbr)
        correct, input_value = validate_generic(self.ent_shelfspace[sku.nbr],
                                     noneg_val=True, int_val=True)
        if correct and sku.param["shelfspace"] != input_value:
            sku.param["shelfspace"] = int(input_value)
            self._reset_kpi_results(sku=sku, target_rol=False)
            sku.redo_ana_calc = True
            sku.redo_sim = True
        return correct

    def _val_conc(self, sku):
        """ Validate the concurrent parameter. """
        self._reset_kpi_results(sku=sku)
        sku.redo_ana_calc = True
        sku.redo_sim = True
        sku.param["concurrent"] = bool(self.conc_var[sku.nbr].get())
        self.highlight_diff(self.chk_conc, "concurrent", sku.nbr)
        return True

    def _reset_redo_sim(self):
        """ Reset the simulation status for all SKUs. """
        for sku in self.skus:
            sku.redo_sim = True

    def _val_maxr(self):
        """ Validate the maximum number of repeats. """
        correct, input_value = validate_generic(self.ent_maxr,
                                     noneg_val=True, nozero_val = True,
                                     int_val=True)
        if correct and input_value < MIN_REPEATS:
            correct = False
            self.ent_maxr.insert(0, "Error (min>max): ")
        if correct and self.sim_param["maxrepeats"] != input_value:
            self.sim_param["maxrepeats"] = int(input_value)
            self._reset_kpi_results(sim_only=True)
            self.new_simoptions = True
            self._reset_redo_sim()
        return correct

    def _val_nrp(self):
        """ Validate the length of the subrun. """
        correct, input_value = validate_generic(self.ent_nrp,
                                     noneg_val=True, nozero_val = True,
                                     int_val=True)
        if correct and self.sim_param["nrperiods"] != input_value:
            self.sim_param["nrperiods"] = int(input_value)
            self._reset_kpi_results(sim_only=True)
            self.new_simoptions = True
            self._reset_redo_sim()
        return correct

    def _val_warmup(self):
        """ Validate the length of the warmup. """
        correct, input_value = validate_generic(self.ent_warmup,
                                     noneg_val=True, nozero_val = True,
                                     int_val=True)
        if correct and self.sim_param["warmup"] != input_value:
            self.sim_param["warmup"] = int(input_value)
            self._reset_kpi_results(sim_only=True)
            self.new_simoptions = True
            self._reset_redo_sim()
        return correct

    def _val_wp(self):
        """ Validate the week pattern parameter. """
        self.sim_param["weekpattern"] = bool(self.wp_var.get())
        self._reset_kpi_results(sim_only=True)
        self.new_simoptions = True
        self._reset_redo_sim()
        return True

    def _val_wf(self, day):
        """ Validate the week fraction. """
        correct, input_value = validate_generic(self.ent_wf[day],
                                     noneg_val=True, ub_1=True)
        if correct and self.week_fractions[day] != input_value:
            # Replace
            self.week_fractions[day] = float(input_value)
            self._reset_kpi_results(sim_only=True)
            self.new_simoptions = True
            self._reset_redo_sim()
        return correct

    # def _val_wlt(self, day):
    #     """ Validate the schedule. """
    #     correct, input_value = validate_generic(self.ent_wlt[day])
    #     if correct and self.wlt[day] != input_value:
    #         self.wlt[day] = float(input_value)
    #         self._reset_kpi_results(sim_only=True)
    #         self.new_simoptions = True
    #         self._reset_redo_sim()
    #     return correct

    def _val_tfr(self):
        """ Validate the target fill rate parameter. """
        correct, input_value = validate_generic(self.ent_tfr,
                                     pos_val=True, ub_1eps=True)
        if correct and self.tfr != input_value:
            self.tfr = input_value
            self._reset_fr_results()
            self.new_tfr = True
        return correct

    def _val_trr(self):
        """ Validate the target ready rate parameter. """
        correct, input_value = validate_generic(self.ent_trr,
                                     pos_val=True, ub_1eps=True)
        if correct and self.trr != float(input_value):
            self.trr = float(input_value)
            self._reset_rr_results()
            self.new_trr = True
        return correct

    def update_statusbar(self, msg):
        """ Update the status bar. """
        self.statusbar.configure(text=msg)

    def _check_sku_param(self, sku):
        # Errors in the parameters?
        correct = True
        for val_func in self.val_functions.values():
            if not val_func(sku):
                correct = False

        fit_text = ""
        if correct and sku.param["distribution"] == "Discrete":
            # Two-moment fit of all distributions possible?
            vtm_fit, fit_text = dobr.validate_discrete_vtm(
                sku.param["mean_perioddemand"],
                sku.param["stdev_perioddemand"],
                sku.param["leadtime"],
                sku.param["reviewperiod"])
            if not vtm_fit:
                correct = False

        if correct and sku.nbr == BASE:
            # Save the new input parameters
            try:
                with open("dobr_defaults.json", "w",
                          encoding="utf8") as fp_def:
                    json.dump(sku.param, fp_def)
            except IOError:
                self.update_statusbar(
                    "IO error while saving input parameters")
        elif not correct:
            # Report on the status bar
            self._reset_kpi_results(sku)
            self.update_statusbar(
                "Parameter error(s) in " + sku.name + " scenario " + fit_text)

        return correct

    def _check_sim_param(self):
        # Errors in the simulation parameters?
        correct = True
        for val_func in self.sim_val_functions.values():
            if not val_func():
                correct = False

        if correct:
            # Save the new simulation parameters
            try:
                with open("dobr_simdefaults.json", "w",
                          encoding="utf8") as fp_def:
                    json.dump(self.sim_param, fp_def)
            except IOError:
                self.update_statusbar(
                    "IO error while saving simulation parameters")
        else:
            # Report on the status bar
            #self._reset_kpi_results(sku)
            self.update_statusbar("Errors in the simulation parameters")

        return correct

    def _renew_system_ana(self, sku):
        """ Renew the the class instance of the inventory system."""

        self.system_ana[sku.nbr] = None
        not_yet = True
        if sku.param["lostsales"]:
            if sku.param["distribution"] == "Discrete":
                not_yet = False
                self.system_ana[sku.nbr] = dobr.InvSysDiscreteLS(
                    sku.param["mean_perioddemand"],
                    sku.param["stdev_perioddemand"],
                    sku.param["leadtime"],
                    stdev_leadtime=sku.param["stdev_leadtime"],
                    reviewperiod=sku.param["reviewperiod"],
                    shelflife=int(sku.param["shelflife"]),
                    ioq=sku.param["ioq"],
                    moq=sku.param["moq"],
                    printerror=False)
        else:
            # Initialize the backorder class for the distribution type
            not_yet = False
            if sku.param["distribution"] == "Discrete":
                self.system_ana[sku.nbr] = dobr.InvSysDiscreteBO(
                    sku.param["mean_perioddemand"],
                    sku.param["stdev_perioddemand"],
                    sku.param["leadtime"],
                    stdev_leadtime=sku.param["stdev_leadtime"],
                    reviewperiod=sku.param["reviewperiod"],
                    ioq=sku.param["ioq"],
                    moq=sku.param["moq"],
                    printerror=False)
            elif sku.param["distribution"] == "Gamma":
                self.system_ana[sku.nbr] = dobr.InvSysGammaBO(
                    sku.param["mean_perioddemand"],
                    sku.param["stdev_perioddemand"],
                    sku.param["leadtime"],
                    stdev_leadtime=sku.param["stdev_leadtime"],
                    reviewperiod=sku.param["reviewperiod"],
                    ioq=sku.param["ioq"],
                    printerror=False)
            elif sku.param["distribution"] == "Normal":
                self.system_ana[sku.nbr] = dobr.InvSysNormalBO(
                    sku.param["mean_perioddemand"],
                    sku.param["stdev_perioddemand"],
                    sku.param["leadtime"],
                    stdev_leadtime=sku.param["stdev_leadtime"],
                    reviewperiod=sku.param["reviewperiod"],
                    ioq=sku.param["ioq"],
                    printerror=False)
        return not_yet

    def _recalc_kpis(self):
        """ Recalculate and display the KPIs. """
        # Recalculate for all skus, if error-free and changed
        calc_time = 0
        any_errors = False
        for sku in self.skus:
            any_errors, calc_time = self._recalc_kpis_sku(sku,
                                                          any_errors, calc_time)

        if not any_errors:
            # If no errors, reset the status line
            calc_time *= 1000
            if calc_time < 0.01:
                calc_time *= 1000
                self.update_statusbar("Calculations finished after"
                                      + f" {calc_time: 5.2f} microseconds")
            else:
                self.update_statusbar("Calculations finished after"
                                      + f" {calc_time: 5.2f} milliseconds")
            # No need to calculate the analytic KPIs again
            self.btn_recalc_kpis["state"] = tk.DISABLED
            # We can output the analytic results to the console
            self.btn_print_ana["state"] = tk.NORMAL
            self.btn_plot_ioh_l["state"] = tk.NORMAL
            self.btn_plot_ioh_rl["state"] = tk.NORMAL
            if self.dash["simulation"]:
                self.btn_plot_ip["state"] = tk.NORMAL
                self.btn_plot_os["state"] = tk.NORMAL
            if self.dash["target_rol"]:
                # Reset the change targets flags
                self.new_tfr = False
                self.new_trr = False

    def _recalc_kpis_sku(self, sku, any_errors, calc_time):
        # Errors in the parameters?
        sku.correct = self._check_sku_param(sku)

        if sku.correct:
            # Reset needed of the inventory system?
            if sku.redo_ana_system:
                start = timer()
                sku.not_yet = self._renew_system_ana(sku)
                calc_time += timer() - start
                if sku.not_yet:
                    # Reset the result column to NOT YET and
                    # report on the status bar
                    self._reset_kpi_results(sku)
                    self.update_statusbar(
                        "Scenario " + sku.name + " is not available yet")
                elif self.system_ana[sku.nbr].dobr_error != 0:
                    # Did the reset encounter (other) errors?
                    sku.correct = False
                    self.system_ana[sku.nbr] = None
                    # Reset the result column and report on the status bar
                    self._reset_kpi_results(sku)
                    self.update_statusbar("Scenario " + sku.name
                        + " returned error: "
                        + dobr.ERROR_CODES[self.system_ana[sku.nr].dobr_error])

            if not sku.correct:
                any_errors = True

            if sku.correct and (not sku.not_yet) and sku.redo_ana_calc:
                # Recalculate the analytic KPIs
                start = timer()
                sku.results_ana = self.system_ana[sku.nbr].calc_kpis(
                    self.kpis_list,
                    sku.param["reorderlevel"],
                    capacity=sku.param["shelfspace"],
                    unitcap=sku.param["unitcap"])
                calc_time += timer() - start
                for kpi in self.kpis_list:
                    col_nr = sku.nbr * self.sku_columns
                    acc_kpi = sku.accuracy(kpi)
                    if (self.dash["simulation"] and acc_kpi == dobr.VAL_APPROX
                        and not sku.redo_sim):
                        # Analytic value in confidence interval simulation?
                        delta = sku.precision(kpi)
                        if (sku.mean_sim(kpi) - delta <= sku.value(kpi)
                                <= sku.mean_sim(kpi) + delta):
                            acc_kpi = dobr.VAL_CI
                    display_value(self.lbl_results[kpi][col_nr],
                                       sku.value(kpi), acc_kpi)

                # Recalculate the target reorder levels?
                if self.dash["target_rol"]:
                    calc_time = self._recalc_rols(sku, calc_time)

                # Reset the flag
                sku.redo_ana_system = False
                sku.redo_ana_calc = False
        return any_errors, calc_time

    def _recalc_rols(self, sku, calc_time):
        # Recalculate the target reorder levels

        # Integer or non-integer reorder levels?
        int_rol = False
        if sku.param["distribution"] == "Discrete":
            # Only integer reorder levels with discrete demand
            int_rol = True

        # Fill rate
        if not self._val_tfr():
            self.rol_result_fr[sku.nbr].configure(text=STR_EMPTYRESULT)
        elif sku.redo_ana_system or self.new_tfr:
            start = timer()
            rol_fr, acc_fr = self.system_ana[sku.nbr].targetfillrate(
                self.tfr, acc=True)
            calc_time += timer() - start
            display_value(self.rol_result_fr[sku.nbr],
                                rol_fr, acc_fr, int_val=int_rol)
        # Ready rate
        if not self._val_trr():
            self.rol_result_rr[sku.nbr].configure(text=STR_EMPTYRESULT)
        elif sku.redo_ana_system or self.new_trr:
            start = timer()
            rol_rr, acc_rr = self.system_ana[sku.nbr].targetreadyrate(
                self.trr, acc=True)
            calc_time += timer() - start
            display_value(self.rol_result_rr[sku.nbr],
                                rol_rr, acc_rr, int_val=int_rol)
        return calc_time

    def _sim_kpis(self):
        # First, check the simulation parameters
        sim_correct = self._check_sim_param()
        if sim_correct:
            self.new_simoptions = False
            # Resimulate for all skus, if error-free and changed
            redo_nbr = -1
            sku_sim_list = []
            for sku in self.skus:
                # Errors in the parameters?
                sku.correct = self._check_sku_param(sku)

                if sku.correct:
                    if sku.redo_sim:
                        # Simulation
                        # Init the results
                        redo_nbr += 1
                        sku.reset_sim(redo_nbr)
                        sku_sim_list.append(dict(sku.param))

            if redo_nbr > -1:
                self.btn_sim_kpis["state"] = tk.DISABLED
                self.btn_print_sim["state"] = tk.DISABLED
                # self.btn_plot_w["state"] = tk.DISABLED

                # Create simulation object
                if self.sim_param["weekpattern"]:
                    sim = dobr_invsim.SimInventory(sku_sim_list, self.sim_param,
                        weekfractions=self.week_fractions)
                else:
                    sim = dobr_invsim.SimInventory(sku_sim_list, self.sim_param)
                # Put simulation object on the queue
                self.sim_queue.put(sim)

    def update_results_sim(self, result_repeat):
        """ Update the SKU data with the last simulation repeat. """

        redo_nbr = -1
        for sku_result in result_repeat:
            redo_nbr += 1
            # Lookup corresponding SKU data
            for sku in self.skus:
                if redo_nbr == sku.redo_nbr:
                    sku.add_sim(sku_result)

    def write_results_sim(self, sku):
        """Write the simulation results to the dashboard."""

        # self.btn_recalc_kpis["state"] = tk.NORMAL
        self.btn_print_sim["state"] = tk.NORMAL
        self.btn_plot_ioh_l["state"] = tk.NORMAL
        self.btn_plot_ioh_rl["state"] = tk.NORMAL
        self.btn_plot_ip["state"] = tk.NORMAL
        self.btn_plot_os["state"] = tk.NORMAL
        # if self.sim_param["reportpmfs"]:
        #     self.btn_plot_tbo["state"] = tk.NORMAL
        #     if self.skus[BASE].param["shelflife"] > 0:
        #         self.btn_plot_w["state"] = tk.NORMAL

        col_nr = sku.nbr * self.sku_columns
        col_ana = 0
        if self.dash["analytic"]:
            col_ana = 1
        for kpi in self.kpis_list:
            if dobr.applicable_kpi(kpi, sku.param):
                if sku.precision(kpi) <= self.sim_param["targetprecision"]:
                    sim_acc = dobr.VAL_EXACT
                else:
                    sim_acc = dobr.VAL_APPROX
                # Write simulation results
                display_value(self.lbl_results[kpi][col_nr+col_ana],
                                   sku.mean_sim(kpi), sim_acc)
                if self.dash["sim_columns"] >= 2:
                    display_precision(self.lbl_results[kpi][col_nr+col_ana+1],
                                       sku.precision(kpi))
                if self.dash["sim_columns"] == 3:
                    if kpi in ("Fillrate", "EST"):
                        sim_acc = dobr.VAL_NOTAPPLIC
                    display_value(self.lbl_results[kpi][col_nr+col_ana+2],
                                       sku.stdev_sim(kpi), sim_acc)
            else:
                sim_acc = dobr.VAL_NOTAPPLIC
                display_value(self.lbl_results[kpi][col_nr+col_ana],
                                   0., sim_acc)
                if self.dash["sim_columns"] >= 2:
                    display_value(self.lbl_results[kpi][col_nr+col_ana+1],
                                       0., sim_acc)
                if self.dash["sim_columns"] == 3:
                    display_value(self.lbl_results[kpi][col_nr+col_ana+2],
                                       0., sim_acc)

        if not self.system_ana[sku.nbr] is None and not sku.redo_ana_calc:
            # Rewrite analytical value
            for kpi in self.kpis_list:
                acc_kpi = sku.accuracy(kpi)
                if acc_kpi == dobr.VAL_APPROX:
                    # Analytic value in confidence interval simulation?
                    delta = sku.precision(kpi)
                    if (sku.mean_sim(kpi) - delta <= sku.value(kpi)
                            <= sku.mean_sim(kpi) + delta):
                        acc_kpi = dobr.VAL_CI
                    display_value(self.lbl_results[kpi][col_nr],
                                        sku.value(kpi), acc_kpi)

    def _print_kpis_ana_all(self):
        """ Print the output of the analytic KPIs of all SKUs."""
        for sku in self.skus:
            if sku.correct and not sku.not_yet:
                dobr.print_kpis_ana(sku.results_ana, sku.param, sku_name=sku.name)

    def print_kpis_sim(self, sku):
        """ Print the simulated KPIs to the console. """

        print(("* DoBr simulation output " + sku.name + " ").ljust(70, "=")+"*")
        dobr.print_data_header(sku.param)
        print(" ".ljust(40)+"Mean         Prec. StDev")
        for kpi in sku.results_sim:
            if dobr.applicable_kpi(kpi, sku.param):
                print_line = (f" {dobr.KPI_CATALOGUE[kpi].ljust(36)} :"
                              + f" {sku.mean_sim(kpi): 10.3f}"
                              + f" \u00B1 {sku.precision(kpi):5.3f}")
                if kpi not in ("Fillrate", "EST"):
                    # Second moment
                    print_line += f" {sku.stdev_sim(kpi): 10.3f}"
                print(print_line)
        print("*".ljust(70, "=")+"*")

    def _print_kpis_sim_all(self):
        """ Print the output of the simulated KPIs of all SKUS."""
        for sku in self.skus:
            if sku.correct:
                self.print_kpis_sim(sku)

    def _plot_ioh_l(self):
        """ Plot the IOH_L distributions."""
        self._plot_series("IOH_L", dobr_invsim.PMF_IOH_L)

    def _plot_ioh_rl(self):
        """ Plot the IOH_RL distributions."""
        self._plot_series("IOH_RL", dobr_invsim.PMF_IOH_RL)

    def _plot_ip_plus(self):
        """ Plot the IP+ distributions. """
        self._plot_series("IP+", dobr_invsim.PMF_IP_PLUS)

    def _plot_os(self):
        """ Plot the order size distributions."""
        self._plot_series("OS", dobr_invsim.PMF_OS, low_bound=1)

    def _plot_series(self, pmf_name, dobr_series_nr, low_bound=0):
        """ Plot the IOH distributions."""
        # Number of series
        labels = []
        series = []
        series_x_range = []
        for sku in self.skus:
            if (self.dash["analytic"] and sku.param["distribution"] == "Discrete"
                    and (not sku.redo_ana_calc)):
                # We have only analytic results for discrete demand
                tmp = (self.system_ana[sku.nbr]
                       .fetch_dist(pmf_name, sku.param["reorderlevel"]))
                if not tmp is None:
                    series.append(tmp)
                    series_x_range.append(pmf_range(tmp, x_min=low_bound))
                    labels.append(pmf_name + " Analytic " + sku.name)
            if (self.dash["simulation"] and sku.report_pmfs and (not sku.redo_sim)):
                labels.append(pmf_name + " Simulation " + sku.name)
                tmp = sku.pmfs_sim[dobr_series_nr,:]
                series.append(tmp)
                series_x_range.append(pmf_range(tmp, x_min=low_bound))

        if len(series) == 0:
            tk.messagebox.showwarning(title="DoBr",
                message="No series to plot")
            return
        # Overall min and max for the x range
        x_range = [min(series_x_range, key=lambda x: x[0])[0],
                   max(series_x_range, key=lambda x: x[1])[1]]
        if x_range[1] - x_range[0] >= MAX_BARS:
            # Check if we do not exceed the number of bars on the x-axis
            tk.messagebox.showwarning(title="DoBr",
                message=f"Too many (>{MAX_BARS}) bars to plot")
            return
        pmfs = np.zeros((x_range[1] - x_range[0] + 1, len(series)))
        serie_nr = 0
        for serie in series:
            pmfs[series_x_range[serie_nr][0]-x_range[0]:series_x_range[serie_nr][1]
                 -x_range[0]+1, serie_nr] = serie[series_x_range[serie_nr][0]:
                                                  series_x_range[serie_nr][1]+1]
            serie_nr += 1
        # Plot
        self._plot_display(pmf_name, x_range, labels, pmfs)

    def _plot_display(self, pmf_name, x_range, labels, pmfs):
        """Generic plot method."""
        # Start plotting
        plt.figure("DoBr " + pmf_name)
        # Bar width depends on number of series
        nr_series = len(labels)
        bar_w = 0.8/(2*nr_series)
        x_pos = np.arange(x_range[0], x_range[1]+1)
        for serie in range(nr_series):
            plt.bar(x_pos+serie*bar_w, pmfs[:, serie], width=bar_w,
                    color=(0, serie/nr_series, 0.3),
                    label=labels[serie])
        if x_range[1] - x_range[0] < 100:
            plt.xticks(x_pos)
        else:
            plt.xticks(np.arange(x_range[0], x_range[1]+1, 5))
        plt.xlabel("X")
        plt.ylabel("PMF")
        plt.legend(loc="best")
        plt.title("Distribution " + pmf_name)
        plt.show()

# Start the dashboard window
if __name__ == "__main__":
    # Setup the master window
    MASTER = tk.Tk()
    # Define a queue to communicate with the thread
    SIMQUEUE = qt()
    # Paint the GUI
    GUIREF = DobrDashboard(MASTER, sim_queue=SIMQUEUE)
    # Init the thread
    TRD = Thread(target=update_cycle, args=(GUIREF, SIMQUEUE,))
    TRD.daemon = True
    TRD.start()
    # Disable resizing
    MASTER.resizable(False, False)
    # Start the event loop for the GUI
    MASTER.mainloop()
