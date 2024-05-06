import numpy as np
import matplotlib.pyplot as plt


# Define the parameters
N = 1000  # Number of points
T = 1.0 / N  # Time interval
t = np.linspace(0.0, 1.0, N, endpoint=False)  # Time vector


# Generate a single cycle of a square wave
square_wave = np.zeros(N)
for i in range(N):
    if i < N // 2:
        square_wave[i] = 1


# Compute the discrete Fourier transform (DFT) coefficients
coefficients = np.fft.fft(square_wave)


# Calculate the frequencies corresponding to the coefficients
frequencies = np.fft.fftfreq(N, T)


# Plot the amplitudes
plt.figure(figsize=(10, 6))
plt.stem(frequencies, np.abs(coefficients) / N, 'b', markerfmt='bo', basefmt=" ")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('DFT of a Square Wave')
plt.grid(True)
plt.show()