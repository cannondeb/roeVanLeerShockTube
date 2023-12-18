
import sod
import Fluxes3

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Exact Solution
    gamma = 1.4
    dustFrac = 0.0
    npts = 256
    t = 0.0005
    left_state = (1E5, 1.0, 0.0)
    right_state = (1E4, 0.125, 0.0)
    geometry = (0., 1., 0.5)

    # VanLeer
    points = 512
    timesteps = 256
    flux1 = 0 # Van Leer
    #flux2 = 1 # Steger-Warming
    flux2 = 2 # Roe

    # left_state and right_state set pressure, density and u (velocity)
    # geometry sets left boundary on 0., right boundary on 1 and initial
    # position of the shock xi on 0.5
    # t is the time evolution for which positions and states in tube should be 
    # calculated
    # gamma denotes specific heat
    # note that gamma and npts are default parameters (1.4 and 500) in solve 
    # function
    _, _, values = sod.solve(left_state=left_state, \
        right_state=right_state, geometry=geometry, t=t, \
        gamma=gamma, npts=npts, dustFrac=dustFrac)

    vLValues = Fluxes3.solve(left_state, right_state, geometry, t, gamma, \
        flux1, points, timesteps)

    roeValues = Fluxes3.solve(left_state, right_state, geometry, t, gamma, \
        flux2, points, timesteps)

    # Finally, let's plot the solutions
    f, axarr = plt.subplots(len(values)-1, sharex=True)

    axarr[0].plot(values['x'], values['p'], linewidth=1.5, color='b')
    axarr[0].plot(vLValues['x'], vLValues['p'], linewidth=0, color='k', marker="x", markersize=5)
    axarr[0].plot(roeValues['x'], roeValues['p'], linewidth=0, color='m', marker="o", markersize=3)
    axarr[0].set_ylabel('pressure')
    axarr[0].set_ylim(0, 11E4)

    axarr[1].plot(values['x'], values['rho'], linewidth=1.5, color='r')
    axarr[1].plot(vLValues['x'], vLValues['rho'],  linewidth=0, color='k', marker="x", markersize=5)
    axarr[1].plot(roeValues['x'], roeValues['rho'], linewidth=0, color='m', marker="o", markersize=3)
    axarr[1].set_ylabel('density')
    axarr[1].set_ylim(0, 1.1)

    axarr[2].plot(values['x'], values['u'], linewidth=1.5, color='g')
    axarr[2].plot(vLValues['x'], vLValues['u'],  linewidth=0, color='k', marker="x", markersize=5)
    axarr[2].plot(roeValues['x'], roeValues['u'], linewidth=0, color='m', marker="o", markersize=3)
    axarr[2].set_ylabel('velocity')

    axarr[3].plot(values['x'], values['mach'], linewidth=1.5, color='y')
    axarr[3].plot(vLValues['x'], vLValues['mach'],  linewidth=0, color='k', marker="x", markersize=5)
    axarr[3].plot(roeValues['x'], roeValues['mach'],  linewidth=0, color='m', marker="o", markersize=3)
    axarr[3].set_ylabel('mach')
    
    plt.suptitle('Shocktube results at t={0}\ndust fraction = {1}, gamma={2}'\
                 .format(t, dustFrac, gamma))
    plt.show()