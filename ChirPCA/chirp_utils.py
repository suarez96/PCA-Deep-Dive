import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
import numpy as np
import os


def _butter_filter(lo_bound, hi_bound, r, order=5):
    """
    Apply the standard scipy butter filter.
    """
    nyquist = 0.5 * r
    l = lo_bound / nyquist
    h = hi_bound / nyquist
    b, a = butter(order, [l, h], btype='band', analog=False)
    return b, a


def bandpass(
        x: np.array,
        r: int = 32000,
        lo_bound: float = 2500,
        hi_bound: float = 6000,
        order: int = 5
):
    """
    Source https://www.allaboutbirds.org/news/do-bird-songs-have-frequencies-higher-than-humans-can-hear/
    Bird calls generally range from 1k-10k. Setting the lo and high bounds to 2500 and 6000 allows most of the range
    to be captured in the bandpass when considering the filter rollof at the boundaries.

    Args:
    x: input audio signal
    r: audio signal sampling rate in hz
    lo_bound: high pass boundary
    hi_bound: low pass boundary
    order: filter_order, higher is sharper and more prone to creating artifacts

    """
    b, a = _butter_filter(lo_bound, hi_bound, r, order=order)
    filtered_data = lfilter(b, a, x)
    return filtered_data


def filter_and_detect_song(
        x: np.array,
        r: int = 32000,
        max_duration: float = 30,
        SNR_threshold: float = 5,
        fade_in_length: float = 0.6,
        noise_length: float = 0.1,
        plot: bool = False,
        **kwargs
):
    """
    To extract a bird call from an audio, perform a bandpass, then calculate the power of initial segment
    n^2 (or a slightly later one, accounting for track fade-in). Calculate power of each subsequent segment
    A_i^2, then calculate signal to noise ratio ${A_i^2/{n_2}$ of each window, when the linear SNR is above
    a certain quantity, say that a signal has been detected.

    Args:
    x: input audio signal
    r: audio signal sampling rate in hz
    SNR_threshold: required decibels for audio to be considered signal
    max_duration: maximum number of seconds to process
    fade_in_length: number of seconds estimated for audio fade in
    noise_length: number of seconds to consider for the noise floor
    plot: whether to create and display the snr plot
    """
    x_crop = x[:max_duration * r]
    d = bandpass(x_crop, r=r, **kwargs)

    # in samples
    num_noise_samples = int(r * noise_length)
    num_fade_in_samples = int(r * fade_in_length)
    SNR_kernel = np.ones(num_noise_samples) / num_noise_samples

    detection = np.convolve(SNR_kernel, d ** 2, mode='same')

    noise_power = detection[num_fade_in_samples + num_noise_samples]

    # convert decibels to linear
    SNR_threshold = 10 ** (SNR_threshold / 10)

    # A^2/n^2
    SNR = detection / noise_power
    if plot:
        plt.title('SNR Plot in DB')
        plt.plot(np.arange(0, d.shape[0] / r, 1 / r), 10 * np.log10(SNR))
        plt.show()
        plt.hist(10 * np.log10(SNR[SNR > SNR_threshold]), label='signal', alpha=0.5, fc='green')
        plt.hist(10 * np.log10(SNR[SNR <= SNR_threshold]), label='noise', alpha=0.5, fc='red')
        plt.legend()
        plt.title('Histogram of SNR')
        plt.show()

    detected_signal = d[SNR > SNR_threshold]

    return detected_signal if len(detected_signal) else np.nan


def load_signal(row, root: str = 'train_audio/'):
    """
    Load the ogg audio file into a numpy array.
    Args:
    root: path to the audio file directory
    """

    fp = os.path.join(root, row.filename)

    d, r = sf.read(fp)

    return {
        'data': d,
        'rate': r,
        'latitude': row.latitude,
        'longitude': row.longitude,
        'SPECIES_CODE': row.primary_label,
        'path': fp
    }


def get_freq_feats(
        x: np.array,
        lo_bound: float = 2500,
        hi_bound: float = 6000,
        num_bins: int = 200
):
    """
    Extract the frequency content of an audio and group them into a fixed set of bins. In this manner, all audios' FFTs
    will have the same frequencies in the x-axis.
    Args:
    x: input array
    lo_bound: lower bound for the band pass
    hi_bound: higher bound for the band pass
    num_bins: number of output features for the fft data
    """

    # get and filter frequency content
    fft = abs(np.fft.fft(x))
    freq = np.fft.fftfreq(x.shape[0], 1 / 32000)

    fft = fft[(freq > lo_bound) & (freq < hi_bound)]
    freq = freq[(freq > lo_bound) & (freq < hi_bound)]

    # bin the frequency contents
    bins = np.linspace(lo_bound, hi_bound, num_bins)
    digitized_idxs = np.digitize(freq, bins)
    binned_freqs = [fft[digitized_idxs == i].mean() for i in range(num_bins)]

    return binned_freqs

