import numpy as np
from photon_canon import Medium, System, Illumination, Detector
from photon_canon.hardware import create_oblique_beams, create_cone_of_acceptance
import multiprocessing as mp
import matplotlib.pyplot as plt
from photon_canon.lut import generate_lut, LUT

mp.set_start_method('spawn', force=True)

def generate():

    # Variable parameters: cm inverse
    mu_s_prime_array = np.arange(0, 5, 0.5)
    mu_s_array = mu_s_prime_array / (1 - g)
    mu_a_array = np.arange(0.5, 5.5, 0.5)

    # Make water medium
    di_water = Medium(n=1.33, mu_s=0, mu_a=0, g=0, desc='di water')
    glass = Medium(n=1.523, mu_s=0, mu_a=0, g=0, desc='glass')

    # Create an illuminator
    lamp = Illumination(create_oblique_beams((0, 1), 40, 1.25))

    # Create detection cone
    detector = Detector(create_cone_of_acceptance(r=1.8, na=1, n=1.33))

    # Start the system
    system = System(di_water, 0.2,  # 2mm
                    glass, 0.017,  # 0.17mm
                    surrounding_n=1.33,
                    illuminator=lamp,
                    detector=(detector, 0)
                    )
    tissue = Medium(n=1.33, mu_s=1, mu_a=1, g=g, desc='tissue')  # Placeholder to update at iteration
    system.add(tissue, d)

    # Generate a photon object (either directly or through the system illumination)
    photon = system.beam(batch_size=n, recurse=recurse)
    simulation_id = generate_lut(system,
                                 tissue,
                                 {'mu_s': mu_s_array, 'mu_a': mu_a_array},
                                 photon,
                                 pbar=True)
    return simulation_id

def show_surfaces(simulation_id):
    lut = LUT(dimensions=['mu_s', 'mu_a'], simulation_id=simulation_id, scale=n)
    X, Y, Z = lut.surface()
    X *= (1 - g)
    fig = plt.figure(figsize=[15, 15])
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_wireframe(Y, X, Z)
    ax.set_title('Monte Carlo')
    ax.set_xlabel("$\mu_a$")
    ax.set_ylabel("$\mu_s'$")
    ax.set_zlabel("$R_m$ (a.u.)")

    fig.savefig(f'lut_{simulation_id}.png')

    plt.close(fig)


if __name__ == '__main__':
    # Define parameter for simulation
    g = 0.9
    d = 0.1  # 1 mm
    n = 100000
    recurse = False
    wl0 = 650

    simulation_id = generate()
    show_surfaces(simulation_id)
