# waveform.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Waveform:
    '''
    Class describing a single waveform. Contains:
    1. Voltage List as Numpy array
    2. Voltage Lump Array as list of numpy arrays
    3. Parameter List
    4. Parent Waveform as Numpy array
    '''
    def __init__(self, voltage_list = None, voltage_lump_array = None, params_list = None, parent_wfm = None) -> None:
        self.voltage_list = voltage_list
        self.voltage_lump_array = voltage_lump_array
        self.params_list = params_list
        self.parent_wfm = parent_wfm
    
    def get_voltage_list(self):
        return self.voltage_list
    
    def get_voltage_lump_array(self):
        return self.voltage_lump_array
    
    def get_params_list(self):
        return self.params_list
    
    def get_parent_wfm(self):
        return self.parent_wfm

    def set_voltage_list(self, voltage_list):
        self.voltage_list = voltage_list
    
    def set_voltage_lump_array(self, voltage_lump_array):
        self.voltage_lump_array = voltage_lump_array
    
    def set_params_list(self, params_list):
        self.params_list = params_list
    
    def set_parent_wfm(self, parent_wfm):
        self.parent_wfm = parent_wfm