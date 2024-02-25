import numpy as np

def fourier_2d(data, frequency_vectors):
    """
    Compute the Fourier basis for the given 2D data and frequency vectors using complex exponentials.
    
    Parameters:
    - data: numpy array of shape (N, 2), where N is the number of data points
    - frequency_vectors: list of 2D frequency vectors for the Fourier basis
    
    Returns:
    - Fourier basis expansion of the data in complex form
    """
    N = data.shape[0]
    fourier_values = []

    # Include the original data
    # fourier_values.append(data)

    for freq_vector in frequency_vectors:
        # Calculate the dot product of frequency vector and data, then compute the complex exponential
        dot_product = np.dot(data, np.array(freq_vector).reshape(2, 1))
        exp_values = np.exp(1j * dot_product)
        
        # Append the complex exponential values
        fourier_values.append(exp_values)

    results = np.concatenate(fourier_values, axis=1)
    
    return results

