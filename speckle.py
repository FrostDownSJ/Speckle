import imageio.v2
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare
import scipy.special as sp
import scipy.stats as st 
# import scipy.integrate as integrate
# from scipy.integrate import dblquad
# import multiprocessing
import statistics
from scipy.signal import convolve2d
from scipy.ndimage import zoom


# ## Functions for intensity distribution


# INPUT: 3D intensity matrix with r, g, b, a values
# OUTPUT : 2D intensity matrix

def convert_3d_to_2d(matrix):
    if len(matrix.shape) == 2:
        return matrix
    ver, hor, _ = matrix.shape
    Im = np.zeros((ver, hor))
    for x in range(ver):
         for y in range(hor):
            r, g, b = matrix[x, y, 0], matrix[x, y, 1], matrix[x, y, 2]
            Im[x, y] = r
    return Im


# In[5]:


# INPUT: TIFF from the camera
# OUTPUT: Display the speckle image

def displaySpeckle(path, Name, cmap='viridis', clim=None, save_as=None):
    """
    Display the intensity matrix with optional color map, color limits, and save option.
    
    Parameters:
        path: path
        Name.
        cmap (str): The color map to use for display.
        clim (tuple): Color limits for the plot (optional).
        save_as (str): File name to save the plot (optional).
    """
    # Display the image using matplotlib
    I = plt.imread(path)
    I = convert_3d_to_2d(I)
    plt.imshow(I, cmap=cmap)
    plt.title(Name)
    plt.colorbar()

    # Set color limits if provided
    if clim:
        plt.clim(clim)

    # Save the figure if a filename is provided
    if save_as:
        plt.savefig(save_as)
    
    # Show the plot
    plt.show()

    # Optionally print some statistics
    print(f"Shape of the intensity matrix: {I.shape}")
    print(f"Max value: {I.max()}")
    print(f"Min value: {I.min()}")


# In[6]:


# INPUT: size of the desired image
# OUTPUT: the simulated speckle with the intensity scaled to 8-bit digital intensity [0,255]
# NOTE: the first line of the code controls the correlation length

def simulated(s):
    random_matrix = np.random.rand(2**6, 2**6) # controls the correlation length

    # Compute the complex exponential
    complex_matrix = np.exp(-2 * np.pi * 1j * random_matrix)

    # Compute the 2D FFT, zero-padded to 2048x2048
    fft_matrix = np.fft.fft2(complex_matrix, s)

    # Shift the zero-frequency component to the center
    shifted_fft_matrix = np.fft.fftshift(fft_matrix)

    # Compute the magnitude and square it
    mag = np.abs(shifted_fft_matrix) ** 2
    mag = (mag/mag.max())*255
    return mag


# In[7]:


# INPUT: speckle intensity matrix (2D or 3D), background intensity matrix (2D or 3D)
# OUTPUT: the 2D intensity matrix with background reduction

def backgroundReduction(speckle, bg):
    try:
        # Check if the speckle is a 3D matrix
        if len(speckle.shape) == 3:
            speckle = convert_3d_to_2d(speckle)
        elif len(speckle.shape) != 2:
            raise ValueError("The speckle matrix must be 2D or 3D.")
            
        if len(bg.shape) == 3:
            bg = convert_3d_to_2d(bg)
        elif len(bg.shape) != 2:
            raise ValueError("The background matrix must be 2D or 3D.")
        
        # Continue with the background reduction process
        reduced_speckle = speckle - bg
        
        return reduced_speckle
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")


# In[8]:


# INPUT: speckle intensity matrix (2D or 3D)
# OUTPUT:  increase the resolution of the speckle intensity matrix by a factor 2, using bicubic interpolation method.
def bicubicInterpolation(Im):
    try:
        # Check if the speckle is a 3D matrix
        if len(Im.shape) == 3:
            speckle = convert_3d_to_2d(speckle)
        elif len(Im.shape) != 2:
            raise ValueError("The speckle matrix must be 2D or 3D.")
        
        img = Image.fromarray((Im).astype(np.uint8))
        width, height = img.size
        new_size = (width * 2, height * 2)

        # Resize the image using bicubic interpolation
        img_resized = img.resize(new_size, Image.BICUBIC)
        img_resized.save('resized_image.tif')
        img_resized = np.array(img_resized)
        return img_resized
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")


# In[9]:


# INPUT:  speckle intensity matrix (2D or 3D)
# OUTPUT: decrease the resolution of the speckle intensity matrix by a factor 2, with 2x2 binning

def binningByFactorOfTwo(Im):
    try:
        # Check if the speckle is a 3D matrix
        if len(Im.shape) == 3:
            Im = convert_3d_to_2d(speckle)
        elif len(Im.shape) != 2:
            raise ValueError("The speckle matrix must be 2D or 3D.")
        kernel = np.ones((2, 2)) / 4

        # Convolve the intensity matrix with the averaging kernel
        convolved_matrix = convolve2d(Im, kernel, mode='valid')

        # Downsample by taking every second element in both dimensions
        binned_matrix = convolved_matrix[::2, ::2]
        plt.imshow(binned_matrix, cmap='grey')
        Im = binned_matrix
        return Im
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")


# In[10]:


# INPUT: intensity matrix (2D or 3D), horizontal start (of a dark region), vertical start, horizontal end, vertical end (in pixels)
# OUTPUT: deduct the pixels that have intensities lower than the the average intensity in a dark region,  this directly returns an 1d array that can be used to plotting
# NOTE: select a region that is though to be dark, then the code find the average over that region. The pixels have intensity lower than that average
# in the entire matrix will be discarded. Aim to work for an image with large dark region, e.g. a Gaussian intensity image
def averageReduction(I, horStart, verStart, horEnd, verEnd):
    try:
        # Check if the speckle is a 3D matrix
        if len(I.shape) == 3:
            I = convert_3d_to_2d(I)
        elif len(I.shape) != 2:
            raise ValueError("The speckle matrix must be 2D or 3D.")
            
        horStart= 0 
        verStart = 0
        horEnd= 100
        verEnd = 100
        ImD = np.zeros((horUp - horStart, verUp - verStart))
        # print(Im)
        for x in range(verStart, verUp):
              for y in range(horStart, horUp):
                    ImD[x][y] = I[x][y]

        threshold = ImD.max()
        print('threshold is', threshold)
        
        cnt = 0
        cnMax = 0
        ver, hor = Im.shape
        for x in range(0, ver):
              for y in range(0, hor):
                    if I[x][y] <= threshold:
                        cnt += 1
                    # if Im[x][y] == 255:
                    #     cnMax += 1
        print('amount of pixels that have an intensity lower than threshold is', cnt)
        
        inp = Im.ravel()

        #Ordering and Cropping
        inp = np.sort(inp)
        inp = inp[cnt+1: inp.size]
        return inp
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")


# In[11]:


# INPUT: intensity matrix (2D or 3D), horizontal start (of selected region), vertical start, horizontal end, vertical end (in pixels)
# OUTPUT: the intensity matrix with the selected size
def pixelSelection(I, hor_start, ver_start, ho_end, ver_end):
    try:
        # Check if the speckle is a 3D matrix
        if len(I.shape) == 3:
            I = convert_3d_to_2d(I)
        elif len(I.shape) != 2:
            raise ValueError("The speckle matrix must be 2D or 3D.")
            
        ver_length = ver_end - ver_start
        hor_length = hor_end - hor_start
        selected = np.zeros((ver_length, hor_length))
        for y in range(0, ver_length):
            for x in range(0, hor_length):
                selected[y][x] = Im[ver_start+y-1][hor_start+x-1]
            
        return selected

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")


# In[12]:


# INPUT： 1d array of intensity
# #OUTPUT: Plot the probability distribution in 1) regular scale 2) logarithmic scale with the theory 
def plotProbabilityDistribution(inp):
    try:
        # Check if the input is 1D
        if len(inp.shape) != 1:
            raise ValueError("The array needs to be 1D")
        
        # Compute necessary statistics
        avg = inp.mean()
        I_0 = inp.max() / avg
        sigma = np.std(inp)
        # print("Average Intensity:", avg)
        # print("Input Array:", inp)

        # Normalize input by average
        inpu = inp / avg
        # print("Max of Normalized Input:", inpu.max())
        # print("Min of Normalized Input:", inpu.min())

        # Histogram of the normalized input
        hist, bin_edges = np.histogram(inpu, bins=70, density=True)

        # Calculate the bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot the histogram (relative probability)
        plt.plot(bin_centers, hist, label='Relative Probability')
        plt.xlabel('Intensity (I/⟨I⟩)')
        plt.ylabel('Relative Probability P(I)')
        plt.title('Relative Probability of Intensity Data')
        plt.legend()
        plt.savefig('Prob.png')
        plt.show()

        # Create an array of intensities for the theoretical distribution
        I_Im = np.linspace(0, inp.max(), 1000)
        P = np.exp(-I_Im / avg)
        I_Im = I_Im / avg

        # Plot the theoretical distribution with logarithmic scale
        plt.figure(figsize=(10, 6))
        plt.plot(I_Im, P, label='$P = e^{-I/\\langle I \\rangle}/\\langle I \\rangle$', color='blue')

        # Actual data plot (histogram)
        plt.plot(bin_centers, hist, label='Actual Data', color='orange')

        # Set the y-axis to a logarithmic scale
        plt.yscale('log')
        plt.xlabel('$I/\\langle I \\rangle$')
        plt.ylabel('Probability $P$')
        plt.title('Logarithmic Plot of the Probability Distribution')

        # Legend and grid
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Calculate the theoretical frequency for the bin centers
        expected_freq = np.interp(bin_centers, I_Im, P)

        # Normalize the theoretical frequency to match the total histogram count
        expected_freq = expected_freq * np.sum(hist) / np.sum(expected_freq)

        # Chi-square test
        chi2, pv = chisquare(hist, f_exp=expected_freq)

        # Show and save the final plot
        plt.savefig('resultsD.png')
        plt.show()

        # Compute contrast and signal-to-noise ratio
        C = sigma / avg
        R = avg / sigma
        print("Contrast is", C)
        print("Signal-to-noise ratio is", R)
        print("Chi-squared value is", chi2, "and p-value is", pv)
        
    except ValueError as ve:
        print(f"ValueError: {ve}")


# ## Functions for ACF

# In[13]:


# INPUT: x-coord array, y-coord array, wavelength, axial distance, x-waist, y-waist
# OUTPUT: the intensity function of Gaussian beam incorporated with the neccessary change for doing a FFT
def I_acf(g, h, L, z, wx, wy):
    hor = 1280
    ver = 1024
    return I0*np.exp(-2*(((g*L*z-hor/2)**2)/(wx)**2+(((h*L*z-ver/2)**2)/(wy)**2)))


# In[14]:


# INPUT: x-coord array, y-coord array, x-waist, y-waist
# OUTPUT: the intensity function of Gaussian beam
def I_normal(g, h, wx, wy):
    hor = 1280
    ver = 1024
    return I0*np.exp(-2*(((g-hor/2)**2)/wx**2+(((h-ver/2)**2)/wy**2)))


# In[15]:


# INPUT: x values (account for number of y values), y values (actual data)
# OUTPUT: difference in 90 percent confidence range in y values
def confidence90(x_values, y_values):
    pdf = y_values/  np.sum(y_values)

    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(pdf)

    # Find the x-values corresponding to the 2.5th percentile and the 97.5th percentile
    length = x_values[np.searchsorted(cdf, 0.95)] - x_values[np.searchsorted(cdf, 0.05)]
    return length


# In[16]:


# INPUT: data
# OUTPUT: Full width half max of the data
def fwhm(y):
    x = np.linspace(0, len(y)-1, len(y))
    half_max = (np.max(y)+np.min(y)) / 2.0
    # nintyfive = (np.max(y)+np.min(y)) * 0.45
    indices_above_threshold = np.where(y >= half_max)[0]

    if len(indices_above_threshold) < 2:
        return 0  # Not enough points above threshold

    left_index = indices_above_threshold[0]
    right_index = indices_above_threshold[-1]

    fwhm_value = x[right_index] - x[left_index]
    return fwhm_value


# In[17]:


# INPUT: speckle intensity matrix (2D or 3D)
# OUTPUT: return the 2D ACF of the speckle
def speckleImgProcessing(mag):
    # Helper function to convert 3D matrix to 2D by extracting the red channel
    if len(mag.shape) == 3:
            mag = convert_to_2d(mag)
    elif mag.ndim != 2:
        raise ValueError("Input matrix must be either 2D or 3D.")

    # Perform FFT operations
    T2 = np.fft.fft2(mag)
    T2 = np.fft.fftshift(T2)
    T3 = np.power(np.abs(T2), 2)
    T4 = np.fft.ifft2(T3)
    T4 = np.fft.fftshift(T4)
    T4 = np.abs(T4)

    return T4


# In[18]:


# INPUT: Measured 2D ACF matrix
# OUTPUT: display full ACF over x and y axis, and zoomed-in ACF (40 pixels around the center) over x and y axis
def displayMeasuredCorreL(T4):
    # Get the shape of the input data
    x = np.arange(T4.shape[1])
    y = np.arange(T4.shape[0])
    
    ver = T4.shape[0]
    hor = T4.shape[1]

    fig, axs = plt.subplots(3, 2, figsize=(12, 15))

    # Top row: Display the full 2D array T4
    im = axs[0, 0].imshow(T4, cmap='viridis', origin='lower', aspect='auto')
    axs[0, 0].set_title('Full 2D ACF')
    axs[0, 0].set_xlabel('Delta x (px)')
    axs[0, 0].set_ylabel('Delta y (px)')
#    axs[0, 0].legend()
    fig.colorbar(im, ax=axs[0, 0])
    axs[0, 1].axis('off')  # Empty subplot to maintain layout symmetry
    
    T4=T4 / T4.max()
    verSlice = T4[:, int(hor / 2)]
    horSlice = T4[int(ver / 2), :]
    
    
    
    # Top left: Full scale plot of ACF over delta x
    axs[1, 0].plot(x, horSlice, label='measured')
    axs[1, 0].set_title('ACF over delta x, full scale')
    axs[1, 0].set_xlabel('Delta x (px)')
    axs[1, 0].set_ylabel('ACF')
    axs[1, 0].legend()
    print('Horizontal FWHM of full wave pattern is', fwhm(horSlice))

    # Top right: Full scale plot of ACF over delta y
    axs[1, 1].plot(y, verSlice, label='measured')
    axs[1, 1].set_title('ACF over delta y, full scale')
    axs[1, 1].set_xlabel('Delta y (px)')
    axs[1, 1].set_ylabel('ACF')
    axs[1, 1].legend()
    print('Vertical FWHM of full wave pattern is', fwhm(horSlice))

    # Bottom left: Zoomed-in plot of ACF over delta x
    axs[2, 0].plot(x, horSlice, label='measured')
    axs[2, 0].set_xlim(int(hor/2)-20, int(hor/2)+20)  # Adjust to zoom in
    # axs[2, 0].set_ylim(0.6, 1.0)  # Adjust to zoom in (example limits)
    axs[2, 0].set_title('ACF over delta x, zoomed in')
    axs[2, 0].set_xlabel('Delta x (px)')
    axs[2, 0].set_ylabel('ACF')
    axs[2, 0].legend()
    print('Horizontal FWHM of cropped wave pattern is', fwhm(horSlice[int(hor/2)-20: int(hor/2)+20]))
    

    # Bottom right: Zoomed-in plot of ACF over delta y
    axs[2, 1].plot(y, verSlice, label='measured')
    axs[2, 1].set_xlim(int(ver/2)-20, int(ver/2)+20)  # Adjust to zoom in
    # axs[2, 1].set_ylim(0.6, 1.0)  # Adjust to zoom in (example limits)
    axs[2, 1].set_title('ACF over delta y, zoomed in')
    axs[2, 1].set_xlabel('Delta y (px)')
    axs[2, 1].set_ylabel('ACF')
    axs[2, 1].legend()
    print('Vertical FWHM of cropped wave pattern is', fwhm(verSlice[int(ver/2)-20: int(ver/2)+20]))

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()


# In[19]:


# INPUT: wavelength of the lightbeam, distance away from the lens, waists of the Gaussian
#           this is based on setup where we have a lens before (or after) the diffuser, and we measure at the focal length
# OUTPUT: theoretical 2D ACF
def TheoreticalACF(L, z, wx, wy):
    hor = 1280
    ver =  1024
    gr = int(hor)
    hr = int(ver)
    gr_range = np.linspace(0, gr-1, hor)
    hr_range = np.linspace(0, hr-1, ver)
    alpha_range = np.linspace(0, hor-1, hor)
    beta_range = np.linspace(0, ver-1, ver)
    g,h = np.meshgrid(gr_range, hr_range)
    # Compute the intensity grid
    intensity_grid = I_acf(g, h, L, z, wx, wy)
    # Shift the zero-frequency component to the center of the spectrum
    fft_intensity = np.fft.fftshift(np.fft.fft2(intensity_grid))
    # Compute the normalized result
    C2I_values = avg**2 * (1 + ((L*z)**2)*np.abs(fft_intensity / np.sum(intensity_grid))**2)
    
    return C2I_values


# In[20]:


# INPUT: Theoretical 2D FFT
# OUTPUT: display full ACF over x and y axis, and zoomed-in ACF (40 pixels around the center) over x and y axis
def displayTheoreticalCorreL(Theo):
        # Get the shape of the input data
    x = np.arange(Theo.shape[1])
    y = np.arange(Theo.shape[0])
    
    ver = Theo.shape[0]
    hor = Theo.shape[1]

    fig, axs = plt.subplots(3, 2, figsize=(12, 15))

    # Top row: Display the full 2D array T4
    im = axs[0, 0].imshow(Theo, cmap='viridis', origin='lower', aspect='auto')
    axs[0, 0].set_title('Full 2D Theoretical ACF')
    axs[0, 0].set_xlabel('Delta x (px)')
    axs[0, 0].set_ylabel('Delta y (px)')
#    axs[0, 0].legend()
    fig.colorbar(im, ax=axs[0, 0])
    axs[0, 1].axis('off')  # Empty subplot to maintain layout symmetry
    
        
    Theo = Theo/ Theo.max()    
    
    verSlice = Theo[:, int(hor / 2)]
    horSlice = Theo[int(ver / 2), :]
    
    
    # Top left: Full scale plot of ACF over delta x
    axs[1, 0].plot(x, horSlice, label='theoretical', color='orange')
    axs[1, 0].set_title('ACF over delta x, full scale')
    axs[1, 0].set_xlabel('Delta x (px)')
    axs[1, 0].set_ylabel('ACF')
    axs[1, 0].legend()
    print('Horizontal FWHM of full wave pattern is', fwhm(horSlice))

    # Top right: Full scale plot of ACF over delta y
    axs[1, 1].plot(y, verSlice, label='theoretical', color='orange')
    axs[1, 1].set_title('ACF over delta y, full scale')
    axs[1, 1].set_xlabel('Delta y (px)')
    axs[1, 1].set_ylabel('ACF')
    axs[1, 1].legend()
    print('Vertical FWHM of full wave pattern is', fwhm(horSlice))

    # Bottom left: Zoomed-in plot of ACF over delta x
    axs[2, 0].plot(x, horSlice, label='theoretical', color='orange')
    axs[2, 0].set_xlim(int(hor/2)-20, int(hor/2)+20)  # Adjust to zoom in
    # axs[2, 0].set_ylim(0.6, 1.0)  # Adjust to zoom in (example limits)
    axs[2, 0].set_title('ACF over delta x, zoomed in')
    axs[2, 0].set_xlabel('Delta x (px)')
    axs[2, 0].set_ylabel('ACF')
    axs[2, 0].legend()
    print('Horizontal FWHM of cropped wave pattern is', fwhm(horSlice[int(hor/2)-20: int(hor/2)+20]))
    

    # Bottom right: Zoomed-in plot of ACF over delta y
    axs[2, 1].plot(y, verSlice, label='theoretical', color='orange')
    axs[2, 1].set_xlim(int(ver/2)-20, int(ver/2)+20)  # Adjust to zoom in
    # axs[2, 1].set_ylim(0.6, 1.0)  # Adjust to zoom in (example limits)
    axs[2, 1].set_title('ACF over delta y, zoomed in')
    axs[2, 1].set_xlabel('Delta y (px)')
    axs[2, 1].set_ylabel('ACF')
    axs[2, 1].legend()
    print('Vertical FWHM of cropped wave pattern is', fwhm(verSlice[int(ver/2)-20: int(ver/2)+20]))

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()


# In[21]:


# INPUT: Measure 2D FFT, theoretical 2D FFT
# OUTPUT: comparison in full ACF over x and y axis, and zoomed-in ACF (40 pixels around the center) over x and y axis
def displayComparison(T4, C2I):
    hor, ver = C2I_values.shape[1], C2I_values.shape[0]
    
    # Theoretical ACF (full scale, x-axis)
    T6 = C2I_values[int(ver / 2)]
    T6 = T6 / T6.max()

    # Measured ACF (full scale, x-axis)
    T7 = T4[int(ver / 2)]
    T7 = T7 / T7.max()

    # Zoomed in range
    zoom_range = slice(int(hor / 2) - 20, int(hor / 2) + 20)
    
    # Zoomed-in ACF (x-axis)
    T8 = T6[zoom_range]
    T9 = T7[zoom_range]

    # Calculating FWHM for theoretical and measured ACFs
    Tfwhm = fwhm(T8)
    Mfwhm = fwhm(T9)

    # Percentage error in FWHM
    percentage_error_x = (np.abs(Tfwhm - Mfwhm) / Tfwhm) * 100

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Full scale plot (x-axis)
    axs[0, 0].plot(T6, label='Theoretical', color='orange')
    axs[0, 0].plot(T7, label='Measured', color='blue')
    axs[0, 0].set_title('ACF over delta x, full scale')
    axs[0, 0].set_xlabel('Delta x (px)')
    axs[0, 0].set_ylabel('ACF')
    axs[0, 0].legend()

    # Zoomed-in plot (x-axis)
    axs[0, 1].plot(T6, label='Theoretical', color='orange')
    axs[0, 1].plot(T7, label='Measured', color='blue')
    axs[0, 1].set_xlim(int(hor/2)-20, int(hor/2)+20)  
    axs[0, 1].set_title('ACF over delta x, zoomed in')
    axs[0, 1].set_xlabel('Delta x (px)')
    axs[0, 1].set_ylabel('ACF')
    axs[0, 1].legend()

    # Repeat the process for the y-axis
    T10 = C2I_values[:, int(hor / 2)]
    T10 = T10 / T10.max()
    
    T11 = T4[:, int(hor / 2)]
    T11 = T11 / T11.max()
    


    T12 = T10[int(ver/2)-20: int(ver/2)+20]
    T13 = T11[int(ver/2)-20: int(ver/2)+20]
    
    Tyfwhm = fwhm(T12)
    Myfwhm = fwhm(T13)
    
    percentage_error_Y = (np.abs(Tyfwhm - Myfwhm) / Tyfwhm) * 100
    
    # Full scale plot (y-axis)
    axs[1, 0].plot(T10, label='Theoretical', color='orange')
    axs[1, 0].plot(T11, label='Measured', color='blue')
    axs[1, 0].set_title('ACF over delta y, full scale')
    axs[1, 0].set_xlabel('Delta y (px)')
    axs[1, 0].set_ylabel('ACF')
    axs[1, 0].legend()
    
    # Zoomed-in plot (y-axis)
    axs[1, 1].plot(T10, label='Theoretical', color='orange')
    axs[1, 1].plot(T11, label='Measured', color='blue')
    axs[1, 1].set_xlim(int(ver/2)-20, int(ver/2)+20)  
    axs[1, 1].set_title('ACF over delta y, zoomed in')
    axs[1, 1].set_xlabel('Delta y (px)')
    axs[1, 1].set_ylabel('ACF')
    axs[1, 1].legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Print results
    print("FWHM for theoretical x-axis:", Tfwhm)
    print("FWHM for measured x-axis:", Mfwhm)
    print("Percentage error for x-axis FWHM:", percentage_error, "%")
    print("FWHM for theoretical y-axis:", Tyfwhm)
    print("FWHM for measured y-axis:", Myfwhm)
    print("Percentage error for y-axis FWHM:", percentage_error, "%")


# In[22]:


# INPUT: Measure 2D FFT, theoretical 2D FFT
# OUTPUT: comparison (with overlapping bottom line) in full ACF over x and y axis, and zoomed-in ACF (40 pixels around the center) over x and y axis
def displayComparisonWithSameAxis(Mea, Theo):
    hor, ver = C2I_values.shape[1], C2I_values.shape[0]
    
    # Theoretical ACF (full scale, x-axis)
    T6 = C2I_values[int(ver / 2)]
    T6 = T6 / T6.max()

    # Measured ACF (full scale, x-axis)
    T7 = T4[int(ver / 2)]
    T7 = T7 / T7.max()

    # Zoomed in range
    zoom_range_x = slice(int(hor / 2) - 20, int(hor / 2) + 20)
    
    # Zoomed-in ACF (x-axis)
    T8 = T6[zoom_range_x]
    T9 = T7[zoom_range_x]

    # Calculating FWHM for theoretical and measured ACFs
    Tfwhm_x = fwhm(T8)
    Mfwhm_x = fwhm(T9)

    # Percentage error in FWHM
    percentage_error_x = (np.abs(Tfwhm_x - Mfwhm_x) / Tfwhm_x) * 100

    # Repeat the process for the y-axis
    T10 = C2I_values[:, int(hor / 2)]
    T10 = T10 / T10.max()
    
    T11 = T4[:, int(hor / 2)]
    T11 = T11 / T11.max()

    # Zoomed-in ACF (y-axis)
    zoom_range_y = slice(int(ver / 2) - 20, int(ver / 2) + 20)
    T12 = T10[zoom_range_y]
    T13 = T11[zoom_range_y]
    
    Tyfwhm = fwhm(T12)
    Myfwhm = fwhm(T13)
    
    percentage_error_y = (np.abs(Tyfwhm - Myfwhm) / Tyfwhm) * 100
    
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Full scale plot (x-axis)
    ax1 = axs[0, 0]
    ax2 = ax1.twinx()
    ax1.plot(T6, label='Theoretical', color='orange')
    ax2.plot(T7, label='Measured', color='blue')
    ax1.set_title('ACF over delta x, full scale')
    ax1.set_xlabel('Delta x (px)')
    ax1.set_ylabel('Theoretical ACF')
    ax2.set_ylabel('Measured ACF')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Zoomed-in plot (x-axis)
    ax3 = axs[0, 1]
    ax4 = ax3.twinx()
    ax3.plot(T6, label='Theoretical', color='orange')
    ax4.plot(T7, label='Measured', color='blue')
    ax3.set_xlim(int(hor/2)-20, int(hor/2)+20)
    ax4.set_xlim(int(hor/2)-20, int(hor/2)+20)
    ax3.set_title('ACF over delta x, zoomed in')
    ax3.set_xlabel('Delta x (px)')
    ax3.set_ylabel('Theoretical ACF')
    ax4.set_ylabel('Measured ACF')
    ax3.legend(loc='upper left')
    ax4.legend(loc='upper right')

    # Full scale plot (y-axis)
    ax5 = axs[1, 0]
    ax6 = ax5.twinx()
    ax5.plot(T10, label='Theoretical', color='orange')
    ax6.plot(T11, label='Measured', color='blue')
    ax5.set_title('ACF over delta y, full scale')
    ax5.set_xlabel('Delta y (px)')
    ax5.set_ylabel('Theoretical ACF')
    ax6.set_ylabel('Measured ACF')
    ax5.legend(loc='upper left')
    ax6.legend(loc='upper right')
    
    # Zoomed-in plot (y-axis)
    ax7 = axs[1, 1]
    ax8 = ax7.twinx()
    ax7.plot(T10, label='Theoretical', color='orange')
    ax8.plot(T11, label='Measured', color='blue')
    ax7.set_xlim(int(ver/2)-20, int(ver/2)+20)
    ax8.set_xlim(int(ver/2)-20, int(ver/2)+20)
    ax7.set_title('ACF over delta y, zoomed in')
    ax7.set_xlabel('Delta y (px)')
    ax7.set_ylabel('Theoretical ACF')
    ax8.set_ylabel('Measured ACF')
    ax7.legend(loc='upper left')
    ax8.legend(loc='upper right')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Print results
    print("FWHM for theoretical x-axis:", Tfwhm_x)
    print("FWHM for measured x-axis:", Mfwhm_x)
    print("Percentage error for x-axis FWHM:", percentage_error_x, "%")
    print("FWHM for theoretical y-axis:", Tyfwhm)
    print("FWHM for measured y-axis:", Myfwhm)
    print("Percentage error for y-axis FWHM:", percentage_error_y, "%")

    return Tfwhm_x, Mfwhm_x, Tyfwhm, Myfwhm


## Functions Ends


