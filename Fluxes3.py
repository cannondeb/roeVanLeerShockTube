import sod

from numpy import linspace, sqrt

def vanLeerP(gamma, rho, c, mach):
    fac = 0.25 * rho * c * (mach + 1)**2
    F0 = fac
    F1 = fac * (2 * c / gamma) * (((gamma - 1) * mach / 2) + 1)
    F2 = fac * ( (2 * c**2 / (gamma**2 - 1)) * (1 + ((gamma - 1) * mach / 2))**2 )

    return (F0, F1, F2)


def vanLeerM(gamma, rho, c, mach):
    fac = -0.25 * rho * c *(mach - 1)**2
    F0 = fac
    F1 = fac * (2 * c / gamma) * (((gamma - 1) * mach / 2) - 1)
    F2 = fac * ( (2 * c**2 / (gamma**2 - 1)) * (1 - ((gamma - 1) * mach / 2))**2 )

    return (F0, F1, F2)

def stegerWarming(lambda1, lambda2, lambda3, gamma, rho, u, c):
    fac = rho / (2 * gamma)
    lam1f0 = lambda1 * 1
    lam1f1 = lambda1 * (u - c)
    lam1f2 = lambda1 * ((0.5 * (u - c)**2) + (0.5 * c**2 * (3 - gamma) / (gamma - 1)))

    lam2f0 = lambda2 * 2 * (gamma - 1)
    lam2f1 = lambda2 * 2 * (gamma - 1) * u
    lam2f2 = lambda2 * 2 * (gamma - 1) * u**2

    lam3f0 = lambda3 * 1
    lam3f1 = lambda3 * (u + c)
    lam3f2 = lambda3 * ((0.5 * (u + c)**2) + (0.5 * c**2 * (3 - gamma) / (gamma - 1)))

    F0 = fac * (lam1f0 + lam2f0 + lam3f0)
    F1 = fac * (lam1f1 + lam2f1 + lam3f1)
    F2 = fac * (lam1f2 + lam2f2 + lam3f2)

    return (F0, F1, F2)


def roe(gamma, rhoL, rhoR, uL, uR, cL, cR, pL, pR):
    # Compute roe averaged values
    rho = sqrt(rhoL*rhoR)

    u = ((uL * sqrt(rhoL)) + (uR * sqrt(rhoR))) / (sqrt(rhoL) + sqrt(rhoR))

    htL = (cL**2 / (gamma - 1)) + (0.5*uL**2)
    htR = (cR**2 / (gamma - 1)) + (0.5*uR**2)
    ht = ((htL * sqrt(rhoL)) + (htR * sqrt(rhoR))) / (sqrt(rhoL) + sqrt(rhoR))

    # Now we can get roe averaged c...
    c = sqrt((ht - (0.5 * u**2)) * (gamma - 1))

    # Left and Right Fluxes
    FL0, FL1, FL2 = stegerWarming(uL - cL, uL, uL + cL, gamma, rhoL, uL, cL)
    FR0, FR1, FR2 = stegerWarming(uR - cR, uR, uR + cR, gamma, rhoR, uR, cR)

    delrho = rhoR - rhoL
    delu = uR - uL
    delp = pR - pL

    AdelU0 = (abs(u - c) * (delp - (rho * c * delu)) / (2 * c**2)) + \
        (abs(u) * (delrho - (delp / c**2))) + \
            (abs(u + c) * (delp + (rho * c * delu)) / (2 * c**2))

    AdelU1 = ((abs(u - c) * (delp - (rho * c * delu)) / (2 * c**2)) * (u - c)) + \
        ((abs(u) * (delrho - (delp / c**2))) * u) + \
             ((abs(u + c) * (delp + (rho * c * delu)) / (2 * c**2)) * (u + c))

    AdelU2 = ((abs(u - c) * (delp - (rho * c * delu)) / (2 * c**2)) * (ht - (c*u))) + \
        ((abs(u) * (delrho - (delp / c**2))) * (u**2 / 2)) + \
             ((abs(u + c) * (delp + (rho * c * delu)) / (2 * c**2)) * (ht + (c*u)))

    F0 = 0.5 * (FL0 + FR0 - AdelU0)
    F1 = 0.5 * (FL1 + FR1 - AdelU1)
    F2 = 0.5 * (FL2 + FR2 - AdelU2)

    return (F0, F1, F2)


def solve(left_state, right_state, geometry, t, gamma, fluxFunc, \
    points, timesteps):
    """ 
    Solve the 1-D shocktube problem using Flux Vector Splitting.
    Parameters:
    ----------
    left_state, right_state: tuple
    A tuple of the state (pressure, density, velocity) on each side of the
    shocktube barrier for the ICs.  In the case of a dusty-gas, the density
    should be the gas density.
    geometry: tuple
        A tuple of positions for (left boundary, right boundary, barrier)
    t: float
        Time to calculate the solution at (i.e. end time marching)
    gamma: float
        Constant specific heat ratio of the fluid.
    fluxFunc int:
        Function used to calculate fluxes
        Available Fxns:
            0  VanLeer
            ...
    points: int
        Number of points in spatial domain
    timesteps: int
        Number of points in temporal domain
    """

    
    xl, xr, xi = geometry       # Unpack geometry
    pl, rhol, ul = left_state   # Unpack left initial conditions
    pr, rhor, ur = right_state  # Unpack right initial conditions
    
    # Discretize Domain
    xs       = linspace(xl, xr, points).tolist()
    
    # Solution Arrays
    rho     = [0.] * points
    rhoU    = [0.] * points
    rhoEt   = [0.] * points

    # Flux Arrays
    Fp       = [(0.,0.,0.)] * points
    Fm       = [(0.,0.,0.)] * points
    Froe     = [(0.,0.,0.)] * points

    # Flux Calculation Quantities
    mach    = 0.
    c       = 0. # Speed of Sound

    # Additional quanitities of interest
    p       = [0.] * points
    u       = [0.] * points

    dx = (xr - xl) / points
    dt = t / timesteps


    # Initialize solution vectors and primitives...
    for i in range(points):
        x = i * dx       
        if x < xi:
            p[i] = pl
            rho[i] = rhol
            rhoU[i] = rhol * ul
            rhoEt[i] = (pl / (gamma - 1)) + (0.5 * rhol * ul**2)
        else:
            p[i] = pr
            rho[i] = rhor
            rhoU[i] = rhor * ur
            rhoEt[i] = (pr / (gamma - 1)) + (0.5 * rhor * ur**2)


    # Time March Solution
    for _ in range(timesteps):
        # Compute Fluxes
        for i in range(points):
            # Put quantities at timelevel n in in terms of calculating flux
            c = sod.sound_speed(gamma, p[i], rho[i], 0)
            mach = u[i] / c
            if 0 == fluxFunc:
                Fp[i] =  vanLeerP(gamma, rho[i], c, mach)
                Fm[i] =  vanLeerM(gamma, rho[i], c, mach)
            elif 1 == fluxFunc:
                ui = u[i]

                lam1P = 0.5 * (ui - c + abs(ui - c))
                lam2P = 0.5 * (ui + abs(ui))
                lam3P = 0.5 * (ui + c + abs(ui + c))

                lam1M = 0.5 * (ui - c - abs(ui - c))
                lam2M = 0.5 * (ui - abs(ui))
                lam3M = 0.5 * (ui + c - abs(ui + c))

                Fp[i] = stegerWarming(lam1P, lam2P, lam3P, gamma, rho[i], u[i], c)
                Fm[i] = stegerWarming(lam1M, lam2M, lam3M, gamma, rho[i], u[i], c)
            elif 2 == fluxFunc:
                if (points - 1) != i:
                    cL = c
                    cR = sod.sound_speed(gamma, p[i+1], rho[i+1], 0)
                    Froe[i] = roe(gamma, rho[i], rho[i + 1], u[i], u[i + 1], cL, cR, p[i], p[i+1])
            
        # Update Solution
        if 2 != fluxFunc:
            for i in range(1,points-1):
                rho[i] = rho[i] - ((dt / dx) * ((Fp[i][0] - Fp[i - 1][0]) + (Fm[i + 1][0] - Fm[i][0])))
                rhoU[i] = rhoU[i] - ((dt / dx) * ((Fp[i][1] - Fp[i - 1][1]) + (Fm[i + 1][1] - Fm[i][1])))
                rhoEt[i] = rhoEt[i] - ((dt / dx) * ((Fp[i][2] - Fp[i - 1][2]) + (Fm[i + 1][2] - Fm[i][2])))
                u[i] = rhoU[i] / rho[i]
                p[i] = (gamma - 1) * (rhoEt[i] - (rho[i]* u[i]**2 / 2)) 
        else:
            for i in range(1, points - 1):
                rho[i] = rho[i] - ((dt / dx) * (Froe[i][0] - Froe[i - 1][0]))
                rhoU[i] = rhoU[i] - ((dt / dx) * (Froe[i][1] - Froe[i - 1][1]))
                rhoEt[i] = rhoEt[i] - ((dt / dx) * (Froe[i][2] - Froe[i - 1][2]))
                u[i] = rhoU[i] / rho[i]
                p[i] = (gamma - 1) * (rhoEt[i] - (rho[i]* u[i]**2 / 2)) 
                

    # Postprocessing
    mach = [0.] * points
    for i in range(points):
        #u[i] = rhoU[i] / rho[i]
        #p[i] = (gamma - 1) * (rhoEt[i] - (rho[i]* u[i]**2 / 2)) 
        mach[i] = u[i] / sod.sound_speed(gamma, p[i], rho[i], 0)




    results_dict = {'x':xs, 'p':p, 'rho':rho, 'u':u, 'mach':mach}

    return results_dict