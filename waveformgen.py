import numpy as np
import matplotlib.pyplot as plt
from waveform import Waveform

class WaveformGenerator:
    '''
    Class that describes a waveform generator object that produces a waveform from a parent waveform and paramater list.
    '''
    def __init__(self, parent_wfm, params_list, update_time_limit, v_levels, set_color = False):
        self.parent_wfm = parent_wfm # [prepulse, midpulse, endpulse]
        self.params_list = params_list
        self.update_time_limit = update_time_limit
        self.high = v_levels[0]
        self.low = v_levels[1]
        self.set_color = set_color
        if set_color == True:
            self.color_ = [np.tile(["blue"], parent_wfm[0].size), np.tile(["royalblue"], parent_wfm[1].size), np.tile(["cornflowerblue"],\
                parent_wfm[2].size)]
    
    def get_parent_wfm(self):
        return self.parent_wfm
    
    def get_params_list(self):
        return self.params_list
    
    def get_update_time_limit(self):
        return self.update_time_limit
    
    def get_v_levels(self):
        return self.high, self.low


    def check_params_list(self):
        params_list_keys = self.params_list.keys()
        pre_len, mid_len, end_len = (self.parent_wfm[0].size, self.parent_wfm[1].size, self.parent_wfm[2].size)
        total_len = pre_len + mid_len + end_len
        if 'p1_low' in params_list_keys:
            # check if p1 is negative or positive. negative adds, positive subtracts.
            if self.params_list['p1_low'] < 0:
                pre_len = pre_len + np.abs(self.params_list['p1_low'])
                mid_len = mid_len + np.abs(self.params_list['p1_low'])
                total_len = total_len + 2*np.abs(self.params_list['p1_low'])
            elif self.params_list['p1_low'] >= 0:
                pre_len = pre_len  - self.params_list['p1_low']
                mid_len = mid_len  - self.params_list['p1_low']
                total_len = total_len - 2*np.abs(self.params_list['p1_low'])
        if 'p1_high' in params_list_keys:
            if self.params_list['p1_high'] < 0:
                pre_len = pre_len + np.abs(self.params_list['p1_high'])
                mid_len = mid_len + np.abs(self.params_list['p1_high'])
                total_len = total_len + 2*np.abs(self.params_list['p1_high'])
            elif self.params_list['p1_high'] >= 0:
                pre_len = pre_len  - self.params_list['p1_high']
                mid_len = mid_len  - self.params_list['p1_high']
                total_len = total_len - 2*np.abs(self.params_list['p1_high'])
        if 'p2_low' in params_list_keys:
            if self.params_list['p2_low'] < 0:
                mid_len = mid_len + np.abs(self.params_list['p2_low'])
                end_len = end_len + np.abs(self.params_list['p2_low'])
                total_len = total_len + 2*np.abs(self.params_list['p2_low'])
            elif self.params_list['p2_low'] >= 0:
                mid_len = mid_len  - self.params_list['p2_low']
                end_len = end_len - self.params_list['p2_low']
                total_len = total_len - 2*np.abs(self.params_list['p2_low'])
        if 'p2_high' in params_list_keys:
            if self.params_list['p2_high'] < 0:
                mid_len = mid_len + np.abs(self.params_list['p2_high'])
                end_len = end_len + np.abs(self.params_list['p2_high'])
                total_len = total_len + 2*np.abs(self.params_list['p2_high'])
            else:
                mid_len = mid_len  - self.params_list['p2_high']
                end_len = end_len - self.params_list['p2_high']
                total_len = total_len - 2*np.abs(self.params_list['p2_high'])
        if 'p1_low_high' in params_list_keys:
            if self.params_list['p1_low_high'] < 0:
                pre_len = pre_len + np.abs(self.params_list['p1_low_high'])
                mid_len = mid_len + np.abs(self.params_list['p1_low_high'])
                total_len = total_len + 2*np.abs(self.params_list['p1_low_high'])
            else:
                pre_len = pre_len - np.abs(self.params_list['p1_low_high'])
                mid_len = mid_len - np.abs(self.params_list['p1_low_high'])
                total_len = total_len - 2*np.abs(self.params_list['p1_low_high'])
        if 'p1_high_low' in params_list_keys:
            if self.params_list['p1_high_low'] < 0:
                pre_len = pre_len + np.abs(self.params_list['p1_high_low'])
                mid_len = mid_len + np.abs(self.params_list['p1_high_low'])
                total_len = total_len + 2*np.abs(self.params_list['p1_high_low'])
            else:
                pre_len = pre_len  - np.abs(self.params_list['p1_high_low'])
                mid_len = mid_len - np.abs(self.params_list['p1_high_low'])
                total_len = total_len - 2*np.abs(self.params_list['p1_high_low'])
        if 'p2_low' in params_list_keys:
            if self.params_list['p2_low'] < 0:
                mid_len = mid_len + np.abs(self.params_list['p2_low'])
                end_len = mid_len + np.abs(self.params_list['p2_low'])
                total_len = total_len + 2*np.abs(self.params_list['p2_low'])
            else:
                mid_len = mid_len  - np.abs(self.params_list['p2_low'])
                end_len = end_len - np.abs(self.params_list['p2_low'])
                total_len = total_len - 2*np.abs(self.params_list['p2_low'])
        if 'p2_high' in params_list_keys:
            if self.params_list['p2_high'] < 0:
                end_len = end_len + np.abs(self.params_list['p2_high'])
                mid_len = mid_len + np.abs(self.params_list['p2_high'])
                total_len = total_len + 2*np.abs(self.params_list['p2_high'])
            else:
                end_len = end_len - np.abs(self.params_list['p2_high'])
                mid_len = mid_len - np.abs(self.params_list['p2_high'])
                total_len = total_len - 2*np.abs(self.params_list['p2_high'])
        if 'p2_high_low' in params_list_keys:
            if self.params_list['p2_high_low'] < 0:
                end_len = end_len + np.abs(self.params_list['p2_high_low'])
                mid_len = mid_len + np.abs(self.params_list['p2_high_low'])
                total_len = total_len + 2*np.abs(self.params_list['p2_high_low'])
            else:
                end_len = end_len - np.abs(self.params_list['p2_high_low'])
                mid_len = mid_len - np.abs(self.params_list['p2_high_low'])
                total_len = total_len - 2*np.abs(self.params_list['p2_high_low'])
        if 'p2_low_high' in params_list_keys:
            if self.params_list['p2_low_high'] < 0:
                mid_len = mid_len + np.abs(self.params_list['p2_low_high'])
                end_len = mid_len + np.abs(self.params_list['p2_low_high'])
                total_len = total_len + 2*np.abs(self.params_list['p2_low_high'])
            else:
                mid_len = mid_len  - np.abs(self.params_list['p2_low_high'])
                end_len = end_len - np.abs(self.params_list['p2_low_high'])
                total_len = total_len - 2*np.abs(self.params_list['p2_low_high'])
        if 'da1_low' in params_list_keys:
            if self.params_list['da1_low'] >= 0:
                # adds a dipole that opposes the trend between the prepulse and midpulse
                total_len = total_len + 2*self.params_list['da1_low']
            else:
                return False
        if 'da1_high' in params_list_keys:
            if self.params_list['da1_high'] >= 0:
                total_len = total_len + 2*self.params_list['da1_high']
            else:
                return False
        if 'da1_low_high' in params_list_keys:
            if self.params_list['da1_low_high'] >= 0:
                total_len = total_len + 2*self.params_list['da1_low_high']
            else:
                return False
        if 'da1_high_low' in params_list_keys:
            if self.params_list['da1_high_low'] >= 0:
                total_len = total_len + 2*self.params_list['da1_high_low']
            else:
                return False
        if 'da2_low' in params_list_keys:
            # adds a dipole that opposes the trend between the midpulse and endpulse
            if self.params_list['da2_low'] >= 0:
                total_len = total_len  + 2*self.params_list['da2_low']
            else:
                return False
        if 'da2_high' in params_list_keys:
            if self.params_list['da2_high'] >= 0:
                total_len = total_len + 2*self.params_list['da2_high']
            else:
                return False
        if 'da2_low_high' in params_list_keys:
            if self.params_list['da2_low_high'] >= 0:
                total_len = total_len + 2*self.params_list['da2_low_high']
            else:
                return False
        if 'da2_high_low' in params_list_keys:
            if self.params_list['da2_high_low'] >= 0:
                total_len = total_len + 2*self.params_list['da2_high_low']
            else:
                return False
        if 'd1_low' in params_list_keys:
            # adds a dipole tuning element to the end of the endpulse (i.e. appends a dipole to the waveform)
            if self.params_list['d1_low'] >= 0:
                total_len = total_len + 2*self.params_list['d1_low']
            else:
                return False
        if 'd1_low_high' in params_list_keys:
            if self.params_list['d1_low_high'] >= 0:
                total_len = total_len + 2*self.params_list['d1_low_high']
            else:
                return False
        if 'd1_high' in params_list_keys:
            if self.params_list['d1_high'] >= 0:
                total_len = total_len + 2*self.params_list['d1_high']
            else:
                return False
        if 'd1_high_low' in params_list_keys:
            key_ = 'd1_high_low'
            if self.params_list[key_] >= 0:
                total_len = total_len + 2*self.params_list[key_]
            else:
                return False
        if 'k1_low' in params_list_keys:
            # adds a balanced pulse-like structure between the prepulse and midpulse
            key_ = 'k1_low'
            if self.params_list[key_] > 1:
                total_len = total_len + 2*self.params_list[key_]
            else:
                return False
        if 'k1_high' in params_list_keys:
            key_ = 'k1_high'
            if self.params_list[key_] > 1:
                total_len = total_len + 2*self.params_list[key_]
            else:
                return False
        if 'k1_high_low' in params_list_keys:
            key_ = 'k1_high_low'
            if self.params_list[key_] > 1:
                total_len = total_len + 2*self.params_list[key_]
            else:
                return False
        if 'k1_low_high' in params_list_keys:
            key_ = 'k1_high_low'
            if self.params_list[key_] > 1:
                total_len = total_len + 2*self.params_list[key_]
            else:
                return False
        if 'k2_low' in params_list_keys:
            # adds a balanced pulse-like structure between the prepulse and midpulse
            key_ = 'k2_low'
            if self.params_list[key_] > 1:
                total_len = total_len + 2*self.params_list[key_]
            else:
                return False
        if 'k2_high' in params_list_keys:
            key_ = 'k2_high'
            if self.params_list[key_] > 1:
                total_len = total_len + 2*self.params_list[key_]
            else:
                return False
        if 'k2_high_low' in params_list_keys:
            key_ = 'k2_high_low'
            if self.params_list[key_] > 1:
                total_len = total_len + 2*self.params_list[key_]
            else:
                return False
        if 'k2_low_high' in params_list_keys:
            key_ = 'k2_high_low'
            if self.params_list[key_] > 1:
                total_len = total_len + 2*self.params_list[key_]
            else:
                return False
        
        # dipole structures have been accounted for now, moving on to gaps...
        if 'ga1' in params_list_keys:
            # adds a gap between the prepulse and midpulse
            key_ = 'ga1'
            if self.params_list[key_] >= 0:
                total_len = total_len + self.params_list[key_]
            else:
                return False
        if 'ga2' in params_list_keys:
            # adds a gap between the midpulse and endpulse
            key_ = 'ga2'
            if self.params_list[key_] >= 0:
                total_len = total_len + self.params_list[key_]
            else:
                return False
        if 'gm2' in params_list_keys:
            # adds a gap in the middle of the midpulse
            key_ = 'gm2'
            if self.params_list[key_] >= 0 :
                total_len = total_len + self.params_list[key_]
            else:
                return False
        if 'gm3' in params_list_keys:
            # adds a gap in the middle of the endpulse
            key_ = 'gm3'
            if self.params_list[key_] >= 0 :
                total_len = total_len + self.params_list[key_]
            else:
                return False
        
        # finally, after everything check the total length
        if total_len > self.update_time_limit:
            return False
        if pre_len < 0 or mid_len < 0 or end_len < 0:
            return False # cannot have pulses with negative lengths
        
        return True
    
    def generate_waveform(self):
        '''
        INPUTS:
        1. self --> takes an instance of self, which contains a parent waveform, and parameter list to be applied
                    to the parent waveform.
        OUTPUTS:
        1. waveform object --> returns a waveform object as defined in the Waveform class -- contains a parent waveform, parameter list,
                                voltage lump array, and voltage list
        '''
        if self.check_params_list() == False:
            return None
        generic_wfm = self.parent_wfm.copy()
        ncolor = self.color_.copy()
        params_list_keys = self.params_list.keys()
        pre_sign, mid_sign, end_sign = (0, 0, 0)
        if self.parent_wfm[0].size > 0:
            pre_sign = np.sign(self.parent_wfm[0][0])
        if self.parent_wfm[1].size > 0:
            mid_sign = np.sign(self.parent_wfm[1][0])
        if self.parent_wfm[2].size > 0:
            end_sign = np.sign(self.parent_wfm[2][0])
        if pre_sign == 0:
            pre_sign = -mid_sign
        if end_sign == 0:
            end_sign = -mid_sign
        # process the dipoles first, and then process the gaps
        if 'p1_low' in params_list_keys:
            # adds a low voltage dipole
            # check if p1 is negative or positive. negative adds, positive subtracts.
            key_ = 'p1_low'
            if self.params_list[key_] < 0:
                new_frames_pre = pre_sign*self.low*np.ones((np.abs(self.params_list[key_]),), dtype= np.int32)
                new_frames_mid = mid_sign*self.low*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames_pre), axis = None, dtype = np.int32)
                generic_wfm[1] = np.concatenate((new_frames_mid, generic_wfm[1]), axis  =None, dtype = np.int32)
                if self.set_color:
                    new_color_pre = np.tile(['lightcoral'], new_frames_pre.size)
                    new_color_mid = np.tile(['lightcoral'], new_frames_mid.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color_pre), axis = None)
                    ncolor[1] = np.concatenate((new_color_mid, ncolor[1]), axis = None)

            elif self.params_list[key_] >= 0:
                generic_wfm[0] = generic_wfm[0][0:-self.params_list[key_]]
                generic_wfm[1] = generic_wfm[1][0:-self.params_list[key_]]
        if 'p1_high' in params_list_keys:
            # adds a high voltage dipole
            # check if p1 is negative or positive. negative adds, positive subtracts.
            key_ = 'p1_high'
            if self.params_list[key_] < 0:
                new_frames_pre = pre_sign*self.high*np.ones((np.abs(self.params_list[key_]),), dtype= np.int32)
                new_frames_mid = mid_sign*self.high*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames_pre), axis = None, dtype = np.int32)
                generic_wfm[1] = np.concatenate((new_frames_mid, generic_wfm[1]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_pre = np.tile(['indianred'], new_frames_pre.size)
                    new_color_mid = np.tile(['indianred'], new_frames_mid.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color_pre), axis = None)
                    ncolor[1] = np.concatenate((new_color_mid, ncolor[1]), axis = None)
            elif self.params_list[key_] >= 0:
                generic_wfm[0] = generic_wfm[0][0:-self.params_list[key_]]
                generic_wfm[1] = generic_wfm[1][0:-self.params_list[key_]]
            
        if 'p1_high_low' in params_list_keys:
            # adds a mixed dipole starting high and ending low
            # check if p1 is negative or positive. negative adds, positive subtracts.
            key_ = 'p1_high_low'
            if self.params_list[key_] < 0:
                new_frames_pre = pre_sign*self.high*np.ones((np.abs(self.params_list[key_]),), dtype= np.int32)
                new_frames_mid = mid_sign*self.low*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames_pre), axis = None, dtype = np.int32)
                generic_wfm[1] = np.concatenate((new_frames_mid, generic_wfm[1]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_pre = np.tile(['indianred'], new_frames_pre.size)
                    new_color_mid = np.tile(['lightcoral'], new_frames_mid.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color_pre), axis = None)
                    ncolor[1] = np.concatenate((new_color_mid, ncolor[1]), axis = None)
            elif self.params_list[key_] >= 0:
                generic_wfm[0] = generic_wfm[0][0:-self.params_list[key_]]
                generic_wfm[1] = generic_wfm[1][0:-self.params_list[key_]]
        if 'p1_low_high' in params_list_keys:
            # adds a mixed dipole starting low and ending high
            # check if p1 is negative or positive. negative adds, positive subtracts.
            key_ = 'p1_low_high'
            if self.params_list[key_] < 0:
                new_frames_pre = pre_sign*self.low*np.ones((np.abs(self.params_list[key_]),), dtype= np.int32)
                new_frames_mid = mid_sign*self.high*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames_pre), axis = None, dtype = np.int32)
                generic_wfm[1] = np.concatenate((new_frames_mid, generic_wfm[1]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_pre = np.tile(['lightcoral'], new_frames_pre.size)
                    new_color_mid = np.tile(['indianred'], new_frames_mid.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color_pre), axis = None)
                    ncolor[1] = np.concatenate((new_color_mid, ncolor[1]), axis = None)
            elif self.params_list[key_] >= 0:
                generic_wfm[0] = generic_wfm[0][0:-self.params_list[key_]]
                generic_wfm[1] = generic_wfm[1][0:-self.params_list[key_]]
        if 'p2_low' in params_list_keys:
            # adds a low voltage dipole between the midpulse and endpulse
            key_ = 'p2_low'
            if self.params_list[key_] < 0:
                new_frames_mid = mid_sign*self.low*np.ones((np.abs(self.params_list[key_]),), dtype= np.int32)
                new_frames_end = end_sign*self.low*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames_mid), axis = None, dtype = np.int32)
                generic_wfm[2] = np.concatenate((new_frames_end, generic_wfm[2]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_mid = np.tile(['salmon'], new_frames_mid.size)
                    new_color_end = np.tile(['salmon'], new_frames_end.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color_mid), axis = None)
                    ncolor[2] = np.concatenate((new_color_end, ncolor[2]), axis = None)
            elif self.params_list[key_] >= 0:
                generic_wfm[1] = generic_wfm[1][0:-self.params_list[key_]]
                generic_wfm[2] = generic_wfm[2][0:-self.params_list[key_]]
        if 'p2_high' in params_list_keys:
            # adds a high voltage dipole between the midpulse and endpulse
            # check if p1 is negative or positive. negative adds, positive subtracts.
            key_ = 'p2_high'
            if self.params_list[key_] < 0:
                new_frames_mid = mid_sign*self.high*np.ones((np.abs(self.params_list[key_]),), dtype= np.int32)
                new_frames_end = end_sign*self.high*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames_mid), axis = None, dtype = np.int32)
                generic_wfm[2] = np.concatenate((new_frames_end, generic_wfm[2]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_mid = np.tile(['tomato'], new_frames_mid.size)
                    new_color_end = np.tile(['tomato'], new_frames_end.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color_mid), axis = None)
                    ncolor[2] = np.concatenate((new_color_end, ncolor[2]), axis = None)

            elif self.params_list[key_] >= 0:
                generic_wfm[1] = generic_wfm[1][0:-self.params_list[key_]]
                generic_wfm[2] = generic_wfm[2][0:-self.params_list[key_]]
        if 'p2_low_high' in params_list_keys:
            # adds a mixed (low, high) dipole between the midpulse and endpulse
            # check if p1 is negative or positive. negative adds, positive subtracts.
            key_ = 'p2_low_high'
            if self.params_list[key_] < 0:
                new_frames_mid = mid_sign*self.low*np.ones((np.abs(self.params_list[key_]),), dtype= np.int32)
                new_frames_end = end_sign*self.high*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames_mid), axis = None, dtype = np.int32)
                generic_wfm[2] = np.concatenate((new_frames_end, generic_wfm[2]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_mid = np.tile(['salmon'], new_frames_mid.size)
                    new_color_end = np.tile(['tomato'], new_frames_end.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color_mid), axis = None)
                    ncolor[2] = np.concatenate((new_color_end, ncolor[2]), axis = None)
            elif self.params_list[key_] >= 0:
                generic_wfm[1] = generic_wfm[1][0:-self.params_list[key_]]
                generic_wfm[2] = generic_wfm[2][0:-self.params_list[key_]]
        if 'p2_high_low' in params_list_keys:
            # adds a mixed (high, low) dipole between the midpulse and endpulse
            key_ = 'p2_high_low'
            if self.params_list[key_] < 0:
                new_frames_mid = mid_sign*self.high*np.ones((np.abs(self.params_list[key_]),), dtype= np.int32)
                new_frames_end = end_sign*self.low*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames_mid), axis = None, dtype = np.int32)
                generic_wfm[2] = np.concatenate((new_frames_end, generic_wfm[2]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_mid = np.tile(['tomato'], new_frames_mid.size)
                    new_color_end = np.tile(['salmon'], new_frames_end.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color_mid), axis = None)
                    ncolor[2] = np.concatenate((new_color_end, ncolor[2]), axis = None)

            elif self.params_list[key_] >= 0:
                generic_wfm[1] = generic_wfm[1][0:-self.params_list[key_]]
                generic_wfm[2] = generic_wfm[2][0:-self.params_list[key_]]
        if 'da1_low' in params_list_keys:
            key_ = 'da1_low'
            if self.params_list[key_] >= 0:
                # adds a dipole that opposes the trend between the prepulse and midpulse
                new_frames_pre = -pre_sign*self.low*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                new_frames_mid = -mid_sign*self.low*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames_pre), axis = None, dtype = np.int32)
                generic_wfm[1] = np.concatenate((new_frames_mid, generic_wfm[1]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_pre = np.tile(['gold'], new_frames_pre.size)
                    new_color_mid = np.tile(['gold'], new_frames_mid.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color_pre), axis = None)
                    ncolor[1] = np.concatenate((new_color_mid, ncolor[1]), axis = None)

        if 'da1_high' in params_list_keys:
            key_ = 'da1_high'
            if self.params_list[key_] >= 0:
                # adds a dipole that opposes the trend between the prepulse and midpulse
                new_frames_pre = -pre_sign*self.high*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                new_frames_mid = -mid_sign*self.high*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames_pre), axis = None, dtype = np.int32)
                generic_wfm[1] = np.concatenate((new_frames_mid, generic_wfm[1]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_pre = np.tile(['darkkhaki'], new_frames_pre.size)
                    new_color_mid = np.tile(['darkkhaki'], new_frames_mid.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color_pre), axis = None)
                    ncolor[1] = np.concatenate((new_color_mid, ncolor[1]), axis = None)

        if 'da1_low_high' in params_list_keys:
            key_ = 'da1_low_high'
            if self.params_list[key_] >= 0:
                new_frames_pre = -pre_sign*self.low*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                new_frames_mid = -mid_sign*self.high*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames_pre), axis = None, dtype = np.int32)
                generic_wfm[1] = np.concatenate((new_frames_mid, generic_wfm[1]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_pre = np.tile(['gold'], new_frames_pre.size)
                    new_color_mid = np.tile(['darkkhaki'], new_frames_mid.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color_pre), axis = None)
                    ncolor[1] = np.concatenate((new_color_mid, ncolor[1]), axis = None)

        if 'da1_high_low' in params_list_keys:
            key_ = 'da1_high_low'
            if self.params_list[key_] >= 0:
                new_frames_pre = -pre_sign*self.high*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                new_frames_mid = -mid_sign*self.low*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames_pre), axis = None, dtype = np.int32)
                generic_wfm[1] = np.concatenate((new_frames_mid, generic_wfm[1]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_pre = np.tile(['darkkhaki'], new_frames_pre.size)
                    new_color_mid = np.tile(['gold'], new_frames_mid.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color_pre), axis = None)
                    ncolor[1] = np.concatenate((new_color_mid, ncolor[1]), axis = None)
        if 'da2_low' in params_list_keys:
            # adds a dipole that opposes the trend between the midpulse and endpulse
            key_ = 'da2_low'
            if self.params_list[key_] >= 0:
                new_frames_mid = -mid_sign*self.low*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                new_frames_end = -end_sign*self.low*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames_mid), axis = None, dtype = np.int32)
                generic_wfm[2] = np.concatenate((new_frames_end, generic_wfm[2]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_mid = np.tile(['orchid'], new_frames_mid.size)
                    new_color_end = np.tile(['orchid'], new_frames_end.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color_mid), axis = None)
                    ncolor[2] = np.concatenate((new_color_end, ncolor[2]), axis = None)

        if 'da2_high' in params_list_keys:
            key_ = 'da2_high'
            if self.params_list[key_] >= 0:
                new_frames_mid = -mid_sign*self.high*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                new_frames_end = -end_sign*self.high*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames_mid), axis = None, dtype = np.int32)
                generic_wfm[2] = np.concatenate((new_frames_end, generic_wfm[2]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_mid = np.tile(['magenta'], new_frames_mid.size)
                    new_color_end = np.tile(['magenta'], new_frames_end.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color_mid), axis = None)
                    ncolor[2] = np.concatenate((new_color_end, ncolor[2]), axis = None)
        if 'da2_low_high' in params_list_keys:
            key_ = 'da2_low_high'
            if self.params_list[key_] >= 0:
                new_frames_mid = -mid_sign*self.low*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                new_frames_end = -end_sign*self.high*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames_mid), axis = None, dtype = np.int32)
                generic_wfm[2] = np.concatenate((new_frames_end, generic_wfm[2]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_mid = np.tile(['orchid'], new_frames_mid.size)
                    new_color_end = np.tile(['magenta'], new_frames_end.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color_mid), axis = None)
                    ncolor[2] = np.concatenate((new_color_end, ncolor[2]), axis = None)
        if 'da2_high_low' in params_list_keys:
            key_ = 'da2_high_low'
            if self.params_list[key_] >= 0:
                new_frames_mid = -mid_sign*self.high*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                new_frames_end = -end_sign*self.low*np.ones((np.abs(self.params_list[key_]), ), dtype = np.int32)
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames_mid), axis = None, dtype = np.int32)
                generic_wfm[2] = np.concatenate((new_frames_end, generic_wfm[2]), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color_mid = np.tile(['magenta'], new_frames_mid.size)
                    new_color_end = np.tile(['orchid'], new_frames_end.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color_mid), axis = None)
                    ncolor[2] = np.concatenate((new_color_end, ncolor[2]), axis = None)
        if 'd1_low' in params_list_keys:
            # adds a dipole tuning element to the end of the endpulse (i.e. appends a dipole to the waveform)
            key_ = 'd1_low'
            if self.params_list[key_]>= 0:
                new_frames = np.tile([end_sign*self.low, -end_sign*self.low], self.params_list[key_])
                generic_wfm[2] = np.concatenate((generic_wfm[2], new_frames), dtype = np.int32, axis = None)
                if self.set_color:
                    new_colors = np.tile(['lime'], new_frames.size)
                    ncolor[2] = np.concatenate((ncolor[2], new_colors), axis = None)
        if 'd1_low_high' in params_list_keys:
            key_ = 'd1_low_high'
            if self.params_list[key_] >= 0:
                new_frames = np.tile([end_sign*self.low, -end_sign*self.high], self.params_list[key_])
                generic_wfm[2] = np.concatenate((generic_wfm[2], new_frames), dtype = np.int32, axis = None)
                if self.set_color:
                    new_colors = np.tile(['lime', 'seagreen'], self.params_list[key_])

                    ncolor[2] = np.concatenate((ncolor[2], new_colors), axis = None)
        if 'd1_high' in params_list_keys:
            key_ = 'd1_high'
            if self.params_list[key_] >= 0:
                new_frames = np.tile([end_sign*self.high, -end_sign*self.high], self.params_list[key_])
                generic_wfm[2] = np.concatenate((generic_wfm[2], new_frames), dtype = np.int32, axis = None)
                if self.set_color:
                    new_colors = np.tile(['seagreen'], new_frames.size)
                    ncolor[2] = np.concatenate((ncolor[2], new_colors), axis = None)
        if 'd1_high_low' in params_list_keys:
            key_ = 'd1_high_low'
            if self.params_list[key_] >= 0:
                new_frames = np.tile([end_sign*self.high, -end_sign*self.low], self.params_list[key_])
                generic_wfm[2] = np.concatenate((generic_wfm[2], new_frames), dtype = np.int32, axis = None)
                if self.set_color:
                    new_colors = np.tile(['seagreen', 'lime'], self.params_list[key_])
                    ncolor[2] = np.concatenate((ncolor[2], new_colors), axis = None)
        if 'k1_low' in params_list_keys:
            # adds a balanced pulse-like structure between the prepulse and midpulse (that opposes the trend between the two pulses)
            key_ = 'k1_low'
            if self.params_list[key_] > 1:
                new_frames_pre = np.tile([-pre_sign*self.low], self.params_list[key_])
                new_frames_mid = np.tile([-np.sign(new_frames_pre[0]*self.low)], self.params_list[key_])
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames_pre), axis = None, dtype = np.int32)
                generic_wfm[1] = np.concatenate((new_frames_mid, generic_wfm[1]), dtype = np.int32, axis = None)
                if self.set_color:
                    new_color_pre = np.tile(['cyan'], new_frames_pre.size)
                    new_color_mid = np.tile(['cyan'], new_frames_mid.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color_pre), axis = None)
                    ncolor[1] = np.concatenate((new_color_mid, ncolor[1]), axis = None)
        if 'k1_high' in params_list_keys:
            key_ = 'k1_high'
            if self.params_list[key_] > 1:
                new_frames_pre = np.tile([-pre_sign*self.high], self.params_list[key_])
                #new_frames_mid = np.tile([-np.sign(new_frames_pre[0]*self.high)], self.params_list[key_])
                new_frames_mid = -new_frames_pre
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames_pre), axis = None, dtype = np.int32)
                generic_wfm[1] = np.concatenate((new_frames_mid, generic_wfm[1]), dtype = np.int32, axis = None)
                if self.set_color:
                    new_color_pre = np.tile(['turquoise'], new_frames_pre.size)
                    new_color_mid = np.tile(['turquoise'], new_frames_mid.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color_pre), axis = None)
                    ncolor[1] = np.concatenate((new_color_mid, ncolor[1]), axis = None)
        if 'k1_high_low' in params_list_keys:
            key_ = 'k1_high_low'
            if self.params_list[key_] > 1:
                new_frames_pre = np.tile([-pre_sign*self.high], self.params_list[key_])
                new_frames_mid = np.tile([-np.sign(new_frames_pre[0]*self.low)], self.params_list[key_])
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames_pre), axis = None, dtype = np.int32)
                generic_wfm[1] = np.concatenate((new_frames_mid, generic_wfm[1]), dtype = np.int32, axis = None)
                if self.set_color:
                    new_color_pre = np.tile(['turquoise'], new_frames_pre.size)
                    new_color_mid = np.tile(['cyan'], new_frames_mid.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color_pre), axis = None)
                    ncolor[1] = np.concatenate((new_color_mid, ncolor[1]), axis = None)
        if 'k1_low_high' in params_list_keys:
            key_ = 'k1_low_high'
            if self.params_list[key_] > 1:
                new_frames_pre = np.tile([-pre_sign*self.low], self.params_list[key_])
                new_frames_mid = np.tile([-np.sign(new_frames_pre[0]*self.high)], self.params_list[key_])
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames_pre), axis = None, dtype = np.int32)
                generic_wfm[1] = np.concatenate((new_frames_mid, generic_wfm[1]), dtype = np.int32, axis = None)
                if self.set_color:
                    new_color_pre = np.tile(['cyan'], new_frames_pre.size)
                    new_color_mid = np.tile(['turquoise'], new_frames_mid.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color_pre), axis = None)
                    ncolor[1] = np.concatenate((new_color_mid, ncolor[1]), axis = None)
        if 'k2_low' in params_list_keys:
            # adds a balanced pulse-like structure between the prepulse and midpulse
            key_ = 'k2_low'
            if self.params_list[key_] > 1:
                new_frames_mid = np.tile([-mid_sign*self.low], self.params_list[key_])
                new_frames_end = -new_frames_mid
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames_mid), axis = None, dtype = np.int32)
                generic_wfm[2] = np.concatenate((new_frames_end, generic_wfm[2]), axis  = None, dtype = np.int32)
                if self.set_color:
                    new_color_mid = np.tile(['plum'], new_frames_mid.size)
                    new_color_end = np.tile(['plum'], new_frames_end.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color_mid), axis = None)
                    ncolor[2] = np.concatenate((new_color_end, ncolor[2]), axis = None)
        if 'k2_high' in params_list_keys:
            key_ = 'k2_high'
            if self.params_list[key_] > 1:
                new_frames_mid = np.tile([-mid_sign*self.high], self.params_list[key_])
                new_frames_end = -new_frames_mid
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames_mid), axis = None, dtype = np.int32)
                generic_wfm[2] = np.concatenate((new_frames_end, generic_wfm[2]), axis  = None, dtype = np.int32)
                if self.set_color:
                    new_color_mid = np.tile(['violet'], new_frames_mid.size)
                    new_color_end = np.tile(['violet'], new_frames_end.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color_mid), axis = None)
                    ncolor[2] = np.concatenate((new_color_end, ncolor[2]), axis = None)
        if 'k2_high_low' in params_list_keys:
            key_ = 'k2_high_low'
            if self.params_list[key_] > 1:
                new_frames_mid = np.tile([-mid_sign*self.high], self.params_list[key_])
                new_frames_end = np.tile([-end_sign*self.low], self.params_list[key_])
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames_mid), axis = None, dtype = np.int32)
                generic_wfm[2] = np.concatenate((new_frames_end, generic_wfm[2]), axis  = None, dtype = np.int32)
                if self.set_color:
                    new_color_mid = np.tile(['violet'], new_frames_mid.size)
                    new_color_end = np.tile(['plum'], new_frames_end.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color_mid), axis = None)
                    ncolor[2] = np.concatenate((new_color_end, ncolor[2]), axis = None)
        if 'k2_low_high' in params_list_keys:
            key_ = 'k2_low_high'
            if self.params_list[key_] > 1:
                new_frames_mid = np.tile([-mid_sign*self.low], self.params_list[key_])
                new_frames_end = np.tile([-end_sign*self.high], self.params_list[key_])
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames_mid), axis = None, dtype = np.int32)
                generic_wfm[2] = np.concatenate((new_frames_end, generic_wfm[2]), axis  = None, dtype = np.int32)
                if self.set_color:
                    new_color_mid = np.tile(['plum'], new_frames_mid.size)
                    new_color_end = np.tile(['violet'], new_frames_end.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color_mid), axis = None)
                    ncolor[2] = np.concatenate((new_color_end, ncolor[2]), axis = None)
        
        # dipole structures have been accounted for now, moving on to gaps...
        if 'ga1' in params_list_keys:
            # adds a gap between the prepulse and midpulse
            key_ = 'ga1'
            if self.params_list[key_] >= 0:
                new_frames = np.tile(np.array([0]), self.params_list[key_])
                generic_wfm[0] = np.concatenate((generic_wfm[0], new_frames), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color = np.tile(["blue"], new_frames.size)
                    ncolor[0] = np.concatenate((ncolor[0], new_color), axis = None)
        if 'ga2' in params_list_keys:
            # adds a gap between the midpulse and endpulse
            key_ = 'ga2'
            if self.params_list[key_] >= 0:
                new_frames = np.tile(np.array([0]), self.params_list[key_])
                generic_wfm[1] = np.concatenate((generic_wfm[1], new_frames), axis = None, dtype = np.int32)
                if self.set_color:
                    new_color = np.tile(["blue"], new_frames.size)
                    ncolor[1] = np.concatenate((ncolor[1], new_color), axis = None)
        if 'gm2' in params_list_keys:
            # adds a gap in the middle of the midpulse
            key_ = 'gm2'
            if self.params_list[key_] >= 0 :
                generic_wfm[1] = np.ndarray.astype(np.insert(generic_wfm[1], np.tile([int(np.floor(generic_wfm[1].size/2))], self.params_list[key_], 0)), dtype = np.int32)
                if self.set_color:
                    ncolor[1] = np.insert(ncolor[1], np.tile([int(np.floor(ncolor[1].size/2))], self.params_list[key_], "blue"))
        if 'gm3' in params_list_keys:
            # adds a gap in the middle of the endpulse
            key_ = 'gm3'
            if self.params_list[key_] >= 0 :
                generic_wfm[2] = np.ndarray.astype(np.insert(generic_wfm[2], np.tile([int(np.floor(generic_wfm[2].size/2))], self.params_list[key_], 0)), dtype = np.int32)
                if self.set_color:
                    ncolor[2] = np.insert(ncolor[2], np.tile([int(np.floor(ncolor[2].size/2))], self.params_list[key_], "blue"))

        generic_wfm[0] = generic_wfm[0].reshape(-1, )
        generic_wfm[1] = generic_wfm[1].reshape(-1, )
        generic_wfm[2] = generic_wfm[2].reshape(-1, )

        voltage_list = np.concatenate(generic_wfm, axis = None, dtype = np.int32)
        new_wav = Waveform(voltage_list = voltage_list, voltage_lump_array=generic_wfm,\
            params_list = self.params_list, parent_wfm=self.parent_wfm)
        return new_wav
    
    def set_parent_wfm(self, ip_from, ip_to):
        # talk to meital to understand
        return
        




                
            


