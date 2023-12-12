import numpy as np

def baseline_drift(X, amplitude=0.2, period_mean=1200, period_std=50):
    # Adds sinusoidal baseline drift to each channel of batch X
    # The Period of this wave is sampled randomly from the Gaussian With Mean and Standard Deviation as Passed to the Method
    # Phase shift is uniformly sampled from [0,2*np.pi]
    # Amplitude is as Passed to the Method
    X_drift = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            period = np.random.normal(period_mean, period_std)
            phase = np.random.uniform(0, 2*np.pi)
            drift = np.zeros(X.shape[2])
            for k in range(X.shape[2]):
                drift[k] = amplitude*np.sin(2*np.pi*k/period + phase)
            X_drift[i][j] = X[i][j] + drift
    return X_drift

def random_noise(X, std=0.02):
    # Adds zero-mean Gaussian random noise to each channel of batch X, with Standard Deviation as Passed to the Method
    return X + np.random.normal(scale=std, size=X.shape)

def burst(X, amplitude=0.05, offset=35):
    # For each sample in the batch X, identifies peaks and if they are sufficiently spaced, places high frequency sinusoid between peaks
    # Offset is the distance between the edges of the region where the high frequency sinusoid is applied and the nearest peak
    # Amplitude is as Passed to the Method
    X_burst = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            burst = np.zeros(X.shape[2])
            delta = np.zeros(int(X.shape[2]/2))
            for k in range(int(X.shape[2]/2)):
                delta[k] = np.abs(X[i][j][2*k+1] - X[i][j][2*k])
            threshold = max(delta) / 4
            peaks = []
            for k in range(int(X.shape[2]/2)):
                if np.abs(X[i][j][2*k+1] - X[i][j][2*k]) > threshold:
                    peaks.append(2*k)
            for k in range(len(peaks)-1):
                if peaks[k+1] - peaks[k] > 4:
                    for m in range(peaks[k]+offset, peaks[k+1]-offset):
                        burst[m] = amplitude*np.sin(np.pi*m/2)
            X_burst[i][j] = X[i][j] + burst
    return X_burst
