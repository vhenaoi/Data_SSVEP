import mne
import numpy as np
import scipy.signal as signal
import linear_filter
from scipy.signal import welch as pwelch   
import matplotlib.pyplot as plt  

def file_set(path, channels, low_fre, high_frec, fs):
    '''Open file .edf and filter'''
    raw = mne.io.read_raw_edf(path,preload=True)
    raw.plot(n_channels=channels, title='EEG Original')
    #raw = raw.filter(low_fre, high_frec, fir_design='firwin', method='fir', filter_length='auto', phase='zero', fir_window='hamming')
    #raw.plot(n_channels=channels, title='EEG Filter')
    #raw.notch_filter(60, fir_design='firwin')
    #raw.plot_psd(tmax=np.inf, average=False)
    data, times = raw[:, :] 
    marks = data[8, :] 
    data2 = linear_filter.eegfiltnew(data, fs, low_fre, high_frec);
    nperseg = fs*20;
    noverlap = int(nperseg/2);
    f, Pxx = pwelch(data, fs, 'hanning', nperseg, noverlap);
    
    plt.subplot(2,1,1);
    plt.plot(f, Pxx[2,:]);
    
    f, Pxx = pwelch(data2, fs, 'hanning', nperseg, noverlap);
    
    plt.subplot(2,1,2);
    plt.plot(f, Pxx[2,:]);
    plt.show();
    data2[8, :] = marks 
    
    return data2, times

def filter_signal(data, fs, low_frec, high_frec):
    data2 = linear_filter.eegfiltnew(data, fs, low_frec, high_frec)
    return data2

def spectrum_power (data_filter, fs, nblock):
    '''Calculate Welch transform '''
    overlap = nblock/2;
    win = signal.hamming(int(nblock),True);        
    f, Pxx = signal.welch(data_filter, fs, window=win, noverlap=overlap, nfft=nblock, return_onesided=True);
    #standarDes = np.std(Pxx, dtype=np.float64)
    #media = np.median(Pxx)
    #PxxS= (Pxx-media)/standarDes
    
    return f, Pxx

def separate_signal (bipolar_signal, time_stimuli, time_rest, fs ):
    '''Separete the signal according to the time '''
    both_eyes = []
    right_eye = []
    left_eye = []
    max_time = time_stimuli*fs - time_rest*fs
    f = lambda A, n=time_stimuli*fs: [A[i:i+n] for i in range(0, len(A), n)]    
    for i in range (0, len(bipolar_signal)):
        signal_split = f(bipolar_signal[i][:])
        right_eye.append(signal_split[0][0:max_time])
        left_eye.append(signal_split[1][0:max_time])
        both_eyes.append(signal_split[2][0:max_time])
    return right_eye, left_eye, both_eyes

def power_band(f, Pxx, frequency):
    bands = []
    for i in range(1, len(Pxx)):   
        for frec in range (0,len(frequency)-1):
            position_low = np.where(f == frequency[frec])[0][0]
            position_high = np.where(f == frequency[frec+1])[0][0]
            total = np.sum(Pxx)
            relative_frec = (np.sum(Pxx[i, position_low:position_high]))/total
            bands.append(relative_frec)
    return bands
    
    
def max_SSVEP(f, P, low_value, high_value):
    
    position = np.where((f >= low_value) & (f <= high_value))
    maxValue = np.max(P[position[0]])
    
    return maxValue

def delete_data (data, times, initial, end, fs):
    deleteL = initial*fs 
    deleteH = len(data) - end*fs 
    data = data[:,deleteL:deleteH]
    times = np.transpose(times)[deleteL:deleteH]
    
    return data, times

    

