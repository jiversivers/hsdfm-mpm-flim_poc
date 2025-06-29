# HSDFM-MPM-FLIM Processing

## MCLUT
powered by `photon_canon.lut`

### Generation
[`hsdfm-mpm-flim_poc/mclut/generate_lut.py`](mclut/generate_lut.py) 

1. Measure actual system hardware.
2. Define simulations of hardware.
3. Instantiate `System` with hardware simulators.
4. Simulate arrays of optical properties (`photon_canon.lut.generate_lut`)

### Validation

#### Round trip
[`hsdfm-mpm-flim_poc/mclut/round_trip_validation.ipynb`](mclut/round_trip_validation.ipynb)
1. The MCLUT is loaded (`photon_canon.lut.LUT`) and a 3$\times$3 Gaussian kernel with $\sigma=2$ is applied to the response surface.
2. Arrays for all 4 parameters are iterated through to find maximum acceptable values for each based on the shape of the surface and bounds of the simualtion.
3. For each parameter, a 32$\times$32 array is drwan from a uniform random distribution $\in [0, P_\mathrm{max})$ accrding to the max values foudn in the previous step.
4. These parameter arrays are passed forward through the model to generate a reflectance curve for each pixel.
5. These reflectance curves are then fit according to the experimental procedure, but $a$ and $b$ are allowed to fit as well, and no baseline shift, $c$ is fit.
6. A $\chi^2_\nu$ of 1.5 is used as a cutoff. The percent of pixels above that threshold is recorded.
7. Each parameter fit was scored where the fits were acceptable.
8. Plots of a linear regression of true vs predicted value were created and saved.

#### Food coloring
[`hsdfm-mpm-flim_poc/phantoms/food_coloring_validation.ipynb`](phantoms/food_coloring_validation.ipynb)
1. Create food coloring solutions of known concentration and absorbance.
2. Image each mixture.
3. Preprocess images as in steps 1 and 2 of [Vascular extraction](#vascular-extraction).
4. Fit the images to an MCLUT (note, scatter properties are fit). 
5. Compare results to known concentration and saturation.

### Post-processing
`photon_canon.lut`
1. Scale outputs by number of input photons.
2. Apply 3$\times$3 with $\sigma=2$ Gaussian blur to the surface, with reflection at the edges.
3. Interp-/Extrap-olate with a regular grid interpolator (`scipy.interpolate`) using cubic interpolation

## HSDFM
[`hsdfm-mpm-flim_poc/animals/basic_hsdfm.py`](animals/basic_hsdfm.py)

### Vascular extraction
powered by `hsdfmpm.hsdfm.fit`

1. Load standard, background, and data images
2. Normalize all to integration time $ I_{s, i, j}' = I_{s, i, j} / \tau_s $
3. Resize to 256$\times$256 pixels.
4. Normalize data to standard and background $ I''_{s, i, j} = \frac{I'_{s, i, j} - I'^{(bg)}_{s, i , j}} {I'^{(std)}_{s, i, j} - I'^{(bg)}_{s, i, j}} $
5. Perform a naive least squares fit of the model $\ln\left(R\right) \sim \sum_iC_i\epsilon_i + b$ over the absorption dominated wavelengths, $\lambda \in [500, 600]$, to calculate a hemoglobin index (`numpy.linalg.lstsq`).
6. Apply a bank of multiscale, rotated Gabor filters to the hemoglobin index image
7. Create two masks 
   1. Apply Otsu's threshold
   2. Apply 11 $\times$ 11 adaptive Gaussian threshold to the Gabor response.
      1. Perform morphological opening in a 3$\times$3 neighborhood.
8. Make a vascular mask from the union of the two masks. ($A \cup B$)
9. Apply the mask to the image.

### Fitting
powered by `hsdfmpm.hsdfm.fit`

1. Fit a simple model in hihg-scatter, low-absorbing portion of the spectrum, $\lambda \in [610, 720]$, to obtain a correction factor $c$: $R_m \sim f(10, 1, 0, 0) + c$ 
2. Create a model wih a MCLUT with $a$ and $b$ fixed at experimentally confirmed values of 10 and 1 and an intercept, $c$ added for baseline adjustment, $R_m - c \sim f(10, 1, \mathrm{[tHb]}, \mathrm{sO_2})$ where $f(a, b, \mathrm{[tHb]}, \mathrm{sO_2}) = \mathrm{MCLUT}(a \frac{\lambda}{\lambda_0}^{-b}, \sum_iC_i\epsilon_i)$
2. Fit the image for $\mathrm{[tHb]}$ and $\mathrm{sO_2}$ using nonlinear least squares (`scipy.optimize.least_sqaures`) with the Trust Region Reflective method to minimize the sum of squared residuals. A finite difference approximation is used for the Jacobian for each parameter w.r.t. the reflectance.
3. Calculate reduced chi squared $\chi^2_\nu = \frac{1}{\nu} \sum_i\frac{(O_i -E_i) ^ 2}{\sigma_i^2}$ where $\nu = P - N$.
4. Calculate the average, standard deviation, and count of $\mathrm{sO_2}$ and $\mathrm{[tHb]}$ where $\chi^2_\nu \leq 1.5$.
5. Create and save vascular masks, maps of output parameters, and colorized figures.

## MPM

### Autofluorescence Intensity
[`hsdfm-mpm-flim_poc/animals/basic_orr.py`](animals/basic_orr.py) powered by `hsdfmpm.mpm.af`.

#### Processing
1. Load 755 and 855 excitation images with associated metadata and daily laser power measurements into `hsdfmpm.mpm.af.OpticalRedoxRatio` object. Under the hood, this...
   1. Performs a least squares fit (`numpy.linalg.lstsq`) of the reference attenuation settings to the measured power to obtain the estimated imaging power.
   2. Applies the instrument-specific transfer function with PMT gain and normalized power.
2. Resize $\mathrm{ORR}$ map to 256$\times$256.
3. Calculate the average, standard deviation, and count of $\mathrm{ORR}$.
4. Create and save a map of the $\mathrm{ORR}$ and a colorized figure.

#### Vascular mapping

#### Vascular mapping from
[`hsdfm-mpm-flim_poc/animals/basic_orr.py`](animals/basic_orr.py)
1. Normalized maps of $\mathrm{NADH}$ intensity are extracted from $\mathrm{ORR}$ objects.
2. A flooding segmentation algorithm (`skimage.segmentation.flood`) is used to mask vasculature with a seed pixel at the minimum intensity value. A tolerance was used that ensures no pixels with $\mathrm{NADH}$ intensity greater than $0.15\, \mu M$-equivalent are considered. 
3. The resultant flood-filled mask is used inverted, to calculate the Euclidean distance (`scipy.ndimage.distance_transform_edt`) of all non-vascular space to the nearest vasculature.
4. Distances and $I_\mathrm{NAD(P)H}$ within the non-vascular regions are fit to a nested model using `scipy.optimize.curve_fit`: 

   $$
      \mathrm{pO_2}\left(\mathrm{sO_2}\right) = \left(\frac{\mathrm{sO_2p_{50}^n}}{1 - \mathrm{sO_2}}\right)^{1/n}
   $$

   $$
      \mathrm{pO_2}\left(d\right) = \mathrm{pO_2^{(0)}}\exp\left(-\frac{d}{d_{\mathrm{limit}}}\right)
   $$
   $$
      \mathrm{NAD(P)H}\left(\mathrm{pO_2}\right) = A_2 - \frac{A1 + A2}{1 + \exp\left({\frac{\mathrm{pO_2} - \mathrm{p_{50}}}{dx}}\right)}
   $$
   $$
      \mathrm{NAD(P)H}\left(d\right) \sim \mathrm{NAD(P)H}\left(\mathrm{pO_2}\left(d\right)\right), \,\,\,  \mathrm{pO_2^{(0)}} = \mathrm{pO_2}\left(\mathrm{sO_2^{\mathrm{(HSDF)}}}\right)
   $$

5. The resulting fit parameters are stored, and the fit quality assessed using $\chi^2_\nu$, where $\sigma$ is estimated using a wavelet estimation for the image (`skimage.restoration.estimate_sigma`).

### Fluorescence Lifetime Imaging
powered by `hsdfmp.mpm.flim`

#### Phasor round-trip validation
[`hsdfm-mpm-flim_poc/animals/package/validating_phasors.py`](package/validating_phasors.py)
1. Arrays of $\tau_1$, $\tau_2$, $\bar{\alpha_1}$, and $\sigma_{\alpha_1}$ were generated in biologically relevant ranges.
2. An array of size 64$\times$64 of $\alpha_1$ values were drawn from a normal distribtuion about $\bar\alpha$ with standard deviation $\sigma_{\alpha_1}$.
3. The random alphas were used to generate a two-species decay with $\tau_1$ and $\tau_2$ lifetime mixtures by drawing $n$ photons from a multionomial distribution with probabilities described by the normalized weighted average of lifetime decays:
$$
   P_i = \frac{\alpha_1 \exp{-t_i/\tau_1} + \alpha_1 \exp{-t_i/\tau_1}}{\sum_i\left(\alpha_1 \exp{-t_i/\tau_1} + \alpha_1 \exp{-t_i/\tau_1}\right)}
$$
4. The decay is either convolved with the actual IRF or not. In the case it is, the correction will be applied later, as calculated from the IRF.
5. The decay is then processed as an experimental image.
6. Scores for retrieved parameters and goodness of fit statistics are stored.

#### Processing
[`hsdfm-mpm-flim_poc/animals/basic_flim.py`](animals/basic_flim.py)

1. Load the IRF file into an `hsdfmpm.mpm.flim.InstrumentResponseFunction` object with appropriate reference lifetime, in this case $\tau_{\mathrm{ref}} = 0\, \mathrm{ ns}$. Also, metadata for the imaging setup is implicitly loaded from the decay file, including time bin number and width. Laser frequency is also input, in this case $f = 80\,\mathrm{MHz}$. Under the hood, this handles...
   1. Performing the phasor transformation on the IRF decay data.
   2. Converting the input reference lifetime into a reference based on laser frequency.
   3. Calculating the complex correction factor, $C_f$, for the IRF decay, $C_f = \frac{P_\mathrm{ref}}{P_\mathrm{meas}}$.
2. Load the sample decay data into a `hsdfmpm.mpm.flim.LifetimeImage` object, with metadata (same as step 1).
3. Load the IRF into the object for correction application.
4. Resize the spatial scale of the decay to 256$\times$256.
5. Get the phasor coordinates from the object. Under the hood...
   1. Calculates the discreet transform: $G = \frac{\sum_i I_i  \cos{\omega * t}}{\sum_iI_i}$ and $S = \frac{\sum_i I_i  \sin{\omega * t}}{\sum_iI_i}$.
   2. Converts the coordinates to a single complex number ($P = G + Si$)
   3. Applies the IRF $C_f$: $P' = C_fP$.
   4. Extracts the real and imaginary components of the corrected phasor.
   5. Applies any thresholding on photon counts (in this case, $\sum_iI_i > 25\mathrm{photons}$).
   6. Applies any median filter passes. In this case, 1 pass with a 5$\times$5 kernel.
6. Fit a line to the phasor cloud using total least squares
   1. The fit is performed using `scipy.ord`. The fit parameters and variance estimates are captured.
   2. Perform singular value decomposition (`numpy.linalg.svd`) on the phasor cloud normalized to variance estimates $G' = \frac{G - \bar{G}}{\sigma_G}$ and $S' = \frac{S - \bar{S}}{\sigma_S}$ to determine the ratio of singular values (the aspect ratio of the underlying data).
   3. The ratio of singular values is used to determine the elliptical nature of the cloud. The $F$ statistic is calculated, $F_{\text{obs}} = \frac{(N-2)\,\Sigma}{N}$ where $\Sigma$ is the aspect ratio and $N$ the number of points. When the probability of this $F \le 0.05$, the null hypothesis of $\Sigma = 0$ was rejected, the fit was considered to support a linear combination of two-species.
7. The intersection of this line with the universal circle is calculated by solving  the quadratic equation: $x = \frac{-\left(2 m b - 1\right) \pm \sqrt{\left(2 m b - 1\right) ^ 2 - 4b^2 \left(m^2 + 1\right)}}{2\left(m^2 + 1\right)}$
8. The lifetimes of the intersection points are found, making up the short and long lifetimes:  $\frac{1}{\omega}\tan\left(\arctan\left(\frac{s}{g}\right)\right)$. Note, the use of the cancelling trigonometric functions protects against $g=0$ and to preserve quadrant information in order to ensure correct signage.
9. All points in the phasor cloud are projected onto this line.
10. The projection locations are used to determine $\alpha_i$ for all points by comparing the distance, $d_i$, along the line from each pure lifetime (where the line and circle intersect): $\alpha_1 = \frac{d_2}{d1 + d2}$
11. These values are used to compute a point-wise mean lifetime, $\tau_m = \alpha_1\tau_1 + \alpha_2\tau_2$.
12. Maps of $\alpha$ and $\tau_m$ are saved, and colorized figures are created and saved.


[//]: # (   $$)

[//]: # (       \mathrm{Var}\left[ G_\mathrm{noisy} \right] = \frac{\sum_i N_i\left&#40;\cos \left&#40; \omega t_i \right&#41; - G\right&#41;^2}{\left&#40; \sum_i N_i \right&#41;^2})

[//]: # (   $$)

[//]: # (   )
[//]: # (   $$ )

[//]: # (       \mathrm{Var}\left[ S_\mathrm{noisy} \right] = \frac{\sum_i N_i\left&#40;\sin \left&#40; \omega t_i \right&#41; - S\right&#41;^2}{\left&#40; \sum_i N_i \right&#41;^2})

[//]: # (   $$)

[//]: # ()
[//]: # (   $$)

[//]: # (       \mathrm{Cov}\left[ G_\mathrm{noisy}, S_\mathrm{noisy} \right] = \frac{\sum_i N_i\left&#40;\cos \left&#40; \omega t_i \right&#41; - G\right&#41; \left&#40;\sin \left&#40; \omega t_i \right&#41; - S\right&#41;}{\left&#40; \sum_i N_i \right&#41;^2})

[//]: # (   $$)