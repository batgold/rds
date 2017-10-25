from pylab import *
from rtlsdr import *

sdr = RtlSdr()

sdr.sample_rate = 2.048e6   #Hz
sdr.center_freq = 99.1e6    #Hz
sdr.freq_correction = 60    #PPM
sdr.gain = 'auto'

print(sdr.read_samples(512))

samples = sdr.read_samples(256*1024)

psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
xlabel('Frequency (MHz)')
ylabel('Relative power (dB)')

show()
