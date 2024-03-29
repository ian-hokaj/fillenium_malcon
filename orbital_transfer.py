import importlib
import sys
# from urllib.request import urlretrieve

# Install drake (and underactuated).
# if 'google.colab' in sys.modules and importlib.util.find_spec('underactuated') is None:
#     urlretrieve(f"http://underactuated.csail.mit.edu/scripts/setup/setup_underactuated_colab.py",
#                 "setup_underactuated_colab.py")
#     from setup_underactuated_colab import setup_underactuated
#     setup_underactuated(underactuated_sha='845157815a58bb51e2033b9c27f235df688e23f6', drake_version='0.25.0', drake_build='releases')

# Setup matplotlib backend (to notebook, if possible, or inline).
# from underactuated.jupyter import setup_matplotlib_backend
# plt_is_interactive = setup_matplotlib_backend()


# python libraries
import numpy as np
import matplotlib.pyplot as plt

# pydrake imports
from pydrake.all import (Variable, SymbolicVectorSystem, DiagramBuilder,
                         LogOutput, Simulator, ConstantVectorSource,
                         MathematicalProgram, Solve, SnoptSolver, PiecewisePolynomial)

# increase default size matplotlib figures
from matplotlib import rcParams
rcParams['figure.figsize'] = (8, 5)

# dictionary of functions to convert the units of the problem data
# the fist argument is the numeric value we want to convert
# the second argument is the unit power
# e.g., m = 2 and power = 2 for square meters
unit_converter = {
    'mass':   lambda m, power=1 : m / 1e3 ** power, # takes kilos, returns tons
    'length': lambda l, power=1 : l / 1e11 ** power, # takes meters, returns hundreds of gigameters
    'time':   lambda t, power=1 : t / (60 * 60 * 24 * 30 * 12)  ** power, # takes seconds, returns years
}


# simple class to store the rocket data
class Rocket(object):

    def __init__(
        self,
        mass,           # mass of the rocket in kg
        thrust_limit,   # max norm of the thrust in kg * m * s^-2
        velocity_limit  # max norm of the velocity in m * s^-1
    ):

        # store mass using the scaled units
        self.mass = unit_converter['mass'](mass)

        # store thrust limit converting the units one by one
        thrust_units = [('mass', 1), ('length', 1), ('time', -2)]
        for (quantity, power) in thrust_units:
            thrust_limit = unit_converter[quantity](thrust_limit, power)
        self.thrust_limit = thrust_limit

        # store velocity limit converting the units one by one
        velocity_units = [('length', 1), ('time', -1)]
        for (quantity, power) in velocity_units:
            velocity_limit = unit_converter[quantity](velocity_limit, power)
        self.velocity_limit = velocity_limit

# instantiate the rocket
rocket = Rocket(
    5.49e5, # mass of Falcon 9 in kg
    .25,    # very small thrust limit in kg * m * s^-2
    170,    # very small velocity limit in m * s^-1
    # 20,
    # 20000,
)


# each planet/asteroid in the problem must be an instance of this class
class Planet(object):

    def __init__(
        self,
        name,          # string with the name of the planet
        color,         # color of the planet for plots
        mass,          # mass of the planet in kg
        position,      # position of the planet in the 2d universe in m
        orbit,         # radius of the orbit in m
        radius=np.nan, # radius of the planet in m (optional)
    ):

        # store the data using the scaled units
        self.name = name
        self.mass = unit_converter['mass'](mass)
        self.position = unit_converter['length'](position)
        self.radius = unit_converter['length'](radius)
        self.orbit = unit_converter['length'](orbit)
        self.color = color

# planet Earth: https://en.wikipedia.org/wiki/Earth
earth = Planet(
    'Earth',                # name of the planet
    'green',                # color for plot
    5.972e24,               # mass in kg
    np.array([2.25e11, 0]), # (average) distance wrt Mars in m
    2e10,                   # orbit radius in m (chosen "big enough" for the plots)
    6.378e6,                # planet radius in m
)

# planet Mars: https://en.wikipedia.org/wiki/Mars
mars = Planet(
    'Mars',      # name of the planet
    'red',       # color for plot
    6.417e23,    # mass in kg
    np.zeros(2), # Mars is chosen as the origin of our 2D universe
    1.5e10,      # orbit radius in m
    3.389e6,     # radius in m
)

# asteroids with random data in random positions
np.random.seed(0)
n_asteroids = 10
asteroids = []
for i in range(n_asteroids):
    mass = np.abs(np.random.randn()) * 5e22
    orbit = mass / 5e12
    earth_from_mars = unit_converter['length'](earth.position, -1)
    asteroid_from_mars = np.random.randn(2) * 3e10 + earth_from_mars / 2
    asteroids.append(
        Planet(
            f'Asteroid_{i}',    # name of the planet
            'brown',            # color for plot
            mass,               # mass in kg
            asteroid_from_mars, # distance from Mars in m
            mass / 5e12,        # radius danger area in m
        )
    )


# main class of the notebook
# it collects the rocket, the planets, and all the asteroids
# implements utility functions needed to write the trajopt
class Universe(object):

    def __init__(
        self,
        rocket, # instance of Rocket
        planets # list of instances of Planet
    ):

        # store data
        self.rocket = rocket
        self.planets = planets

        # gravitational constant in m^3 * kg^-1 * s^-2
        self.G = 6.67e-11

        # gravitational constant in the scaled units
        G_units = [('length', 3), ('mass', -1), ('time', -2)]
        for (quantity, power) in G_units:
            self.G = unit_converter[quantity](self.G, power)

    # given the planet name, returns the Planet instance
    def get_planet(self, name):

        # loop through the planets in the universe
        for planet in self.planets:
            if planet.name == name:
                return planet

        # in case no planet has the given name
        print(name + ' is not in the Universe!')

    # computes  2D distance vector between the rocket and a planet,
    # given the rocket state and the planet name
    def position_wrt_planet(self, state, name):

        # rocket position wrt to the planet position
        planet = self.get_planet(name)
        p = state[:2] - planet.position

        return p

    # computes the rocket acceleration due to a planet
    def acceleration_from_planet(self, state, name):

        # distance from the planet
        p = self.position_wrt_planet(state, name)
        d = p.dot(p) ** .5

        # 2d acceleration vector
        planet = self.get_planet(name)
        a = - self.G * planet.mass / d ** 3  * p

        return a

    # right-hand side of the rocket continuous-time dynamics
    # in the form state_dot = f(state, thrust)
    # (thrust is a 2D vector with the horizontal and vertical thrusts)
    def rocket_continuous_dynamics(self, state, thrust):

        # thrust acceleration
        a = thrust / self.rocket.mass

        # accelerations due to the planets
        for planet in self.planets:
            a = a + self.acceleration_from_planet(state, planet.name)

        # concatenate velocity and acceleration
        state_dot = np.concatenate((state[2:], a))

        return state_dot

    # residuals of the rocket discrete-time dynamics
    # if the vector of residuals is zero, then this method's
    # arguments verify the discrete-time dynamics
    # (implements the implicit Euler integration scheme:
    # https://en.wikipedia.org/wiki/Backward_Euler_method)
    def rocket_discrete_dynamics(self, state, state_next, thrust, time_step):

        # continuous-time dynamics evaluated at the next time step
        state_dot = self.rocket_continuous_dynamics(state_next, thrust)

        # implicit-Euler state update
        residuals = state_next - state - time_step * state_dot

        return residuals

    # helper function for the trajopt problem
    # if the vector of residuals is zero, then the state of
    # the rocket belongs to the desired orbit of the given planet
    # (i.e.: the rocket is on the given orbit, with zero radial
    # velocity, and zero radial acceleration)
    def constraint_state_to_orbit(self, state, planet_name):

        # unpack state, rocket position in relative coordinates
        planet = self.get_planet(planet_name)
        p = state[:2] - planet.position
        v = state[2:]

        # constraint on radial distance
        # sets x^2 + y^2 to the orbit radius squared
        residual_p = p.dot(p) - planet.orbit ** 2

        # radial velocity must be zero
        # sets the time derivative of x^2 + y^2 to zero
        residual_v = p.dot(v)

        # radial acceleration must be zero with zero input
        # sets the second time derivative of x^2 + y^2 to zero
        # why this extra constraint?
        # knowing that the radial velocity is zero is not enough
        # the tangential velocity must be such that the gravitational
        # force is balanced by the centrifugal force
        a = self.acceleration_from_planet(state, planet_name)
        residual_a = p.dot(a) + v.dot(v)

        # gather constraint residuals
        residuals = np.array([residual_p, residual_v, residual_a])

        return residuals

    # bonus method! (not actually needed in the trajopt...)
    # computes the gravity acceleration on the surface of a planet
    def gravity_on_planet_surface(self, name):

        # retrieve planet
        planet = self.get_planet(name)
        if planet is not None:

            # if planet radius is not available
            if np.isnan(planet.radius):
                print(name + ' has unknown radius.')
                return

            # use Newton's law of universal gravitation
            g = self.G * planet.mass / planet.radius ** 2

            # use the converter the other way around
            # to express g in MKS
            g_inverse_units = [('length', -1), ('time', 2)]
            for (quantity, power) in g_inverse_units:
                g = unit_converter[quantity](g, power)

            # print the result
            print('Gravity acceleration on ' + name + f' is {g} m/s^2.')


# instantiate universe
planets = [earth, mars] + asteroids
universe = Universe(rocket, planets)


universe.gravity_on_planet_surface('Earth')
universe.gravity_on_planet_surface('Mars')
universe.gravity_on_planet_surface('Jupiter')




# helper function that plots a circle centered at
# the given point and with the given radius
def plot_circle(center, radius, *args, **kwargs):

    # discretize angle
    angle = np.linspace(0, 2*np.pi)

    # plot circle
    plt.plot(
        center[0] + radius * np.cos(angle),
        center[1] + radius * np.sin(angle),
        *args,
        **kwargs
    )

# function that draws the state-space trajectory of the rocket
# including the planets and the asteroids
def plot_state_trajectory(trajectory, universe):

    for planet in universe.planets:

        # plot planets
        plt.scatter(*planet.position, s=100, c=planet.color)
        plt.text(*planet.position, planet.name)

        # plot orbits
        if not np.isnan(planet.orbit):
            if planet.name == 'Asteroid_1':
                orbit_label = 'Asteroid danger area'
            elif planet.name[:8] == 'Asteroid':
                orbit_label = None
            else:
                orbit_label = planet.name + ' orbit'
            plot_circle(
                planet.position,
                planet.orbit,
                label=orbit_label,
                color=planet.color,
                linestyle='--'
            )

    # plot rocket trajectory
    plt.plot(trajectory.T[0], trajectory.T[1], color='k', label='Rocket trajectory')
    plt.scatter(trajectory[0,0], trajectory[0,1], color='k')

    # misc settings
    length_unit = unit_converter['length'](1)
    plt.xlabel('{:.0e} meters'.format(length_unit))
    plt.ylabel('{:.0e} meters'.format(length_unit))
    plt.grid(True)
    plt.gca().set_aspect('equal')

    # legend
    n_legend = len(plt.gca().get_legend_handles_labels()[0])
    plt.legend(
        loc='upper center',
        ncol=int(n_legend / 2),
        bbox_to_anchor=(.5, 1.25),
        fancybox=True,
        shadow=True
    )
    plt.show()

# function that plots the norm of the rocket thrust and
# velocity normalized on their maximum value
def plot_rocket_limits(rocket, thrust, state):

    # reconstruct time vector
    time_steps = thrust.shape[0]
    time = np.linspace(0, time_steps, time_steps + 1)

    # plot maximum norm limit
    plt.plot(time, np.ones(time_steps + 1), 'r--', label='Limit')

    # plot normalized thrust
    thrust_norm = [np.linalg.norm(t) / rocket.thrust_limit for t in thrust]
    plt.step(time, [thrust_norm[0]] + thrust_norm, label='Thrust / thrust limit')

    # plot normalized velocity
    velocity_norm = [np.linalg.norm(v) / rocket.velocity_limit for v in state[:,2:]]
    plt.plot(time, velocity_norm, label='Velocity / velocity limit')

    # plot limits
    plt.xlim(0, time_steps)
    ymax = max(1, max(thrust_norm), max(velocity_norm)) * 1.05
    plt.ylim(0, ymax)

    # misc settings
    plt.xlabel('Time step')
    plt.grid(True)
    plt.legend()
    plt.show()


# function that interpolates two given positions of the rocket
# velocity is set to zero for all the times
def interpolate_rocket_state(p_initial, p_final, time_steps):
    np.random.seed(0)

    # initial and final time and state
    time_limits = [0., time_steps * time_interval]
    position_limits = np.column_stack((p_initial, p_final))
    state_limits = np.vstack((position_limits, np.zeros((2, 2))))

    # linear interpolation in state
    state = PiecewisePolynomial.FirstOrderHold(time_limits, state_limits)

    # sample state on the time grid and add small random noise
    state_guess = np.vstack([state.value(t * time_interval).T for t in range(time_steps + 1)])
    state_guess += np.random.rand(*state_guess.shape) * 5e-6

    return state_guess




# numeric parameters
time_interval = .5 # in years
time_steps = 100

# initialize optimization
prog = MathematicalProgram()

# optimization variables
state = prog.NewContinuousVariables(time_steps + 1, 4, 'state')
thrust = prog.NewContinuousVariables(time_steps, 2, 'thrust')

# initial orbit constraints
for residual in universe.constraint_state_to_orbit(state[0], 'Earth'):
    prog.AddConstraint(residual == 0)

# terminal orbit constraints
for residual in universe.constraint_state_to_orbit(state[-1], 'Mars'):
    prog.AddConstraint(residual == 0)
    
# discretized dynamics
for t in range(time_steps):
    residuals = universe.rocket_discrete_dynamics(state[t], state[t+1], thrust[t], time_interval)
    for residual in residuals:
        prog.AddConstraint(residual == 0)
    
# initial guess
state_guess = interpolate_rocket_state(
    universe.get_planet('Earth').position,
    universe.get_planet('Mars').position,
    time_steps
)
prog.SetInitialGuess(state, state_guess)


# velocity limits, for all t:
# two norm of the rocket velocity
# lower or equal to the rocket velocity_limit

for t in range(time_steps):
  prog.AddConstraint(state[t][2:4].dot(state[t][2:4]) <= rocket.velocity_limit**2)

# avoid collision with asteroids, for all t, for all asteroids:
# two norm of the rocket distance from the asteroid
# greater or equal to the asteroid orbit

for t in range(time_steps):
  for a in asteroids:
    d = universe.position_wrt_planet(state[t], a.name)
    prog.AddConstraint(d.dot(d) >= a.orbit**2)

# thrust limits, for all t:
# two norm of the rocket thrust
# lower or equal to the rocket thrust_limit
for t in range(time_steps):
  prog.AddConstraint(thrust[t].dot(thrust[t]) <= rocket.thrust_limit**2)

# minimize fuel consumption, for all t:
# add to the objective the two norm squared of the thrust
# multiplied by the time_interval so that the optimal cost
# approximates the time integral of the thrust squared

prog.AddCost(time_interval * sum(t.dot(t) for t in thrust))




# solve mathematical program
solver = SnoptSolver()
result = solver.Solve(prog)

# be sure that the solution is optimal
assert result.is_success()

# retrieve optimal solution
thrust_opt = result.GetSolution(thrust)
state_opt = result.GetSolution(state)

# compute fuel consumption for the optimal trajectory
def fuel_consumption(thrust, time_interval):
    return time_interval * sum(t.dot(t) for t in thrust)
print(f'Is fuel consumption {fuel_consumption(thrust_opt, time_interval)} lower than 250?')


plt.figure()
plot_state_trajectory(state_opt, universe)
plt.show()

plt.figure()
plot_rocket_limits(rocket, thrust_opt, state_opt)
plt.show()