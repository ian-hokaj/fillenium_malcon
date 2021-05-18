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
import matplotlib.animation as animation
import copy

# pydrake imports
from pydrake.all import (Variable, SymbolicVectorSystem, DiagramBuilder,
                         LogOutput, Simulator, ConstantVectorSource,
                         MathematicalProgram, Solve, SnoptSolver, PiecewisePolynomial, eq)

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
    # .25,    # very small thrust limit in kg * m * s^-2
    # 170,    # very small velocity limit in m * s^-1
    .35, #DEBUG
    200, #DEBUG
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

class Asteroid(Planet):

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
        self.uncertainty = 1.03
        diff = self.get_orbit(1) - self.orbit
        self.movex = np.random.randn()*diff
        self.movey = np.random.randn()*diff

    def get_orbit(self, time_step):
        return self.orbit*self.uncertainty**time_step

    def move_step(self):
        # move asteroid somewhere in self.get_orbit(1)
        diff = self.get_orbit(1) - self.orbit
        # favor past movement
        movex = .9*self.movex + .1*np.random.randn()*diff # move anywhere between -diff and diff
        movey = .9*self.movey + .1*np.random.randn()*diff
        self.position += np.array([movex, movey])

# asteroids with random data in random positions
np.random.seed(3) # or 0
n_asteroids = 20
# n_asteroids = 25 #DEBUG
asteroids = []
for i in range(n_asteroids):
    mass = np.abs(np.random.randn()) * 2e22 #5e22
    orbit = mass / 5e12
    earth_from_mars = unit_converter['length'](earth.position, -1)
    asteroid_from_mars = np.random.randn(2) * 3e10 + earth_from_mars / 2
    asteroids.append(
        Asteroid(
            f'Asteroid_{i}',    # name of the planet
            'brown',            # color for plot
            mass,               # mass in kg
            asteroid_from_mars, # distance from Mars in m
            orbit,        # radius danger area in m
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
        for planet in self.planets[:2]: #DEBUG
        # for planet in self.planets:
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

    ### UNCOMMENT TO PLOT OUTPUT TRAJECTORY ###
    # plot rocket trajectory
    # plt.plot(trajectory.T[0], trajectory.T[1], color='k', label='Rocket trajectory')
    # plt.scatter(trajectory[0,0], trajectory[0,1], color='k')

    ### COMMENT THIS TO REVERT ###
    x = trajectory.T[0]
    y = trajectory.T[1]
    line, = ax.plot(x, y, color='k', label='Rocket trajectory')
    def update(num, x, y, line):
        line.set_data(x[:num], y[:num])
        return line,
    ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line],
                              interval=10, blit=True)
    ### END HERE ###
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


# function that plots overall trajectories with movement
def plot_state_trajectory_movement(states, asteroids_over_time, universe):

    for planet in [earth, mars]:

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

    x = states.T[0]
    y = states.T[1]
    line, = ax.plot(x, y, color='k', label='Rocket trajectory')

    orbit_dict = {a.name: plt.Circle((a.position[0], a.position[1]), a.orbit, ec='r', fill=False) for a in asteroids_over_time[0]}

    for v in orbit_dict.values():
        ax.add_patch(v)

    asteroid_x = [a.position[0] for a in asteroids_over_time[0]]
    asteroid_y = [a.position[1] for a in asteroids_over_time[0]]
    scat = ax.scatter(asteroid_x, asteroid_y)


    def update(num, x, y, line, scat):
        line.set_data(x[:num], y[:num])
        asteroid_x = np.array([a.position[0] for a in asteroids_over_time[num]])
        asteroid_y = np.array([a.position[1] for a in asteroids_over_time[num]])
        asteroids_pos = np.vstack((asteroid_x, asteroid_y))
        scat.set_offsets(asteroids_pos.T)
        for a in asteroids_over_time[num]:
            orbit_dict[a.name].center = (a.position[0], a.position[1])
        ret = [line, scat] + [v for v in orbit_dict.values()]
        return ret

    ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line, scat],
                              interval=20, blit=True)
    ### END HERE ###
    plt.show()


# function that plots overall trajectories with movement
def plot_single_window_visual(states, asteroids_at_window, universe):

    for planet in [earth, mars]:

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

    x = states.T[0]
    y = states.T[1]
    line, = ax.plot(x, y, color='k', label='Rocket trajectory')

    orbit_dict = {a.name: plt.Circle((a.position[0], a.position[1]), a.orbit, ec='r', fill=False) for a in asteroids_at_window}

    for v in orbit_dict.values():
        ax.add_patch(v)

    asteroid_x = [a.position[0] for a in asteroids_at_window]
    asteroid_y = [a.position[1] for a in asteroids_at_window]
    scat = ax.scatter(asteroid_x, asteroid_y)


    def update(num, x, y, line, scat):
        line.set_data(x[:num], y[:num])
        for a in asteroids_at_window:
            orbit_dict[a.name].radius = a.get_orbit(num)
        ret = [line, scat] + [v for v in orbit_dict.values()]
        return ret

    ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line, scat],
                              interval=100, blit=True)
    ### END HERE ###
    plt.show()


# function that interpolates two given positions of the rocket
# velocity is set to zero for all the times
def interpolate_rocket_state(p_initial, p_final, time_steps):
    # np.random.seed(0)

    # initial and final time and state
    time_limits = [0., time_steps * time_interval]
    position_limits = np.column_stack((p_initial, p_final))
    state_limits = np.vstack((position_limits, np.zeros((2, 2))))

    # linear interpolation in state
    state = PiecewisePolynomial.FirstOrderHold(time_limits, state_limits)

    # sample state on the time grid and add small random noise
    state_guess = np.vstack([state.value(t * time_interval).T for t in range(time_steps + 1)])
    # state_guess += np.random.rand(*state_guess.shape) * 5e-6

    return state_guess

# rolls out current state dynamics over a horizon or remaining steps (assuming constant thrust)
# Returns state at end of window
def rollout(state, thrust, time_steps, time_interval):
    # coarse dynamics over time steps
    state_dot = universe.rocket_continuous_dynamics(state, thrust)
    final_state = state + state_dot*time_interval*time_steps/2
    state_dot = universe.rocket_continuous_dynamics(final_state, thrust)
    final_state += state_dot*time_interval*time_steps/2
    return final_state

# 4-step rollout dynamics
# def rollout(state, thrust, time_steps, time_interval):
#     # coarse dynamics over time steps
#     state_dot = universe.rocket_continuous_dynamics(state, thrust)
#     final_state = state + state_dot*time_interval*time_steps/4
#     state_dot = universe.rocket_continuous_dynamics(final_state, thrust)
#     final_state += state_dot*time_interval*time_steps/4
#     state_dot = universe.rocket_continuous_dynamics(final_state, thrust)
#     final_state += state_dot*time_interval*time_steps/4
#     state_dot = universe.rocket_continuous_dynamics(final_state, thrust)
#     final_state += state_dot*time_interval*time_steps/4
#     return final_state


def create_prog_for_window(window, start_state, step, total, guess=[], is_initial=False, in_final=False):
    # initialize optimization
    prog = MathematicalProgram()

    # optimization variables
    state = prog.NewContinuousVariables(window + 1, 4, 'state')
    thrust = prog.NewContinuousVariables(window, 2, 'thrust')

    # initial orbit constraints
    if is_initial:
        for residual in universe.constraint_state_to_orbit(state[0], 'Earth'):
            c = prog.AddConstraint(residual == 0)
            c.evaluator().set_description("start in earth orbit")
    else:
        c = prog.AddConstraint(eq(state[0], start_state))
        c.evaluator().set_description("start at last state")


    # terminal orbit constraints
    if in_final:
        for residual in universe.constraint_state_to_orbit(state[-1], 'Mars'):
            c = prog.AddConstraint(residual == 0)
            c.evaluator().set_description("end in mars orbit")

    # discretized dynamics
    for t in range(window):
        residuals = universe.rocket_discrete_dynamics(state[t], state[t+1], thrust[t], time_interval)
        for residual in residuals:
            c = prog.AddConstraint(residual == 0)
            c.evaluator().set_description("dynamics")

    # initial guess
    if is_initial:
        state_guess = interpolate_rocket_state(
            universe.get_planet('Earth').position,
            universe.get_planet('Mars').position,
            total
        )[0:window+1]
    else:
        state_guess = guess[:window+1]
        # state_guess = interpolate_rocket_state(
        #     start_state[0:2],
        #     universe.get_planet('Mars').position,
        #     total-step
        # )[0:window+1]
    prog.SetInitialGuess(state, state_guess)

    # # get closer to mars over this window
    # if not in_final:
    #     p1 = universe.position_wrt_planet(state[0], 'Mars')
    #     d1 = p1.dot(p1) ** .5
    #     p2 = universe.position_wrt_planet(state[-1], 'Mars')
    #     d2 = p2.dot(p2) ** .5
    #     prog.AddConstraint(d2 * 1.05 <= d1)

    # velocity limits, for all t:
    # two norm of the rocket velocity
    # lower or equal to the rocket velocity_limit
    for t in range(window):
      c = prog.AddConstraint(state[t][2:4].dot(state[t][2:4]) <= rocket.velocity_limit**2)
      c.evaluator().set_description("velocity limits")

    # avoid collision with asteroids, for all t, for all asteroids:
    # two norm of the rocket distance from the asteroid
    # greater or equal to the asteroid orbit
    for t in range(window):
      for a in asteroids:
        d = universe.position_wrt_planet(state[t], a.name)
        c = prog.AddConstraint(d.dot(d) >= a.get_orbit(t)**2)
        c.evaluator().set_description("asteroid collision")

    # thrust limits, for all t:
    # two norm of the rocket thrust
    # lower or equal to the rocket thrust_limit
    for t in range(window):
      c = prog.AddConstraint(thrust[t].dot(thrust[t]) <= rocket.thrust_limit**2)
      c.evaluator().set_description("thrust constraints")

    # rollout dynamics (recursive feasibility) constraint
    if not in_final:
        final_state = rollout(state[-1], thrust[-1], total-window-step, time_interval)
        for residual in universe.constraint_state_to_orbit(final_state, 'Mars'):
            c = prog.AddConstraint(residual == 0)
            c.evaluator().set_description("rollout to mars")

    # minimize fuel consumption, for all t:
    # add to the objective the two norm squared of the thrust
    # multiplied by the time_interval so that the optimal cost
    # approximates the time integral of the thrust squared

    prog.AddCost(time_interval * sum(t.dot(t) for t in thrust))

    # solve mathematical program
    solver = SnoptSolver()
    result = solver.Solve(prog)

    # be sure that the solution is optimal
    if not result.is_success():
        for constraint in result.GetInfeasibleConstraints(prog):
            print("violation:", constraint)

    # retrieve optimal solution
    thrust_window = result.GetSolution(thrust)
    state_window = result.GetSolution(state)

    return thrust_window, state_window


# numeric parameters
time_interval = .5 # in years
time_steps = 100
window = 15 # time steps per calculation
# Earth state: [ 2.32035322  0.18721759 -0.04109043  0.01544109]


states = []
thrusts = []
asteroids_movements = []
asteroids_movements.append([copy.deepcopy(a) for a in asteroids])

iter_states = []

next_guess = []

for i in range(time_steps):

    curr_window = min(window, time_steps-i)
    print("ITER", i, "OF", time_steps, "WINDOW", curr_window, "REMAINING", time_steps-curr_window-i)

    # start at previous state, compute over window
    if i == 0:
        thrust_window, state_window = create_prog_for_window(curr_window, None, i, time_steps, is_initial=True)
        states.append(state_window[0])
        # print(states[0])
    elif curr_window < window: # in final approach
        thrust_window, state_window = create_prog_for_window(curr_window, states[-1], i, time_steps, guess=next_guess, in_final=True)
    else:
        thrust_window, state_window = create_prog_for_window(curr_window, states[-1], i, time_steps, guess=next_guess)

    if i % 10 == 0:
        fig, ax = plt.subplots()
        plot_single_window_visual(state_window, asteroids_movements[-1], universe)

    next_guess = state_window[1:]
    next_state = universe.rocket_continuous_dynamics(next_guess[-1], thrust_window[-1])
    next_guess = np.vstack((next_guess, next_state))

    states.append(state_window[1])
    thrusts.append(thrust_window[0])
    asteroids_movements.append([copy.deepcopy(a) for a in asteroids])

    for asteroid in asteroids:
        asteroid.move_step()

    iter_states.append(state_window)

# state_opt = np.array(state_window)
# thrust_opt = np.array(thrust_window)
state_opt = np.array(states)
thrust_opt = np.array(thrusts)
asteroids_movements = np.array(asteroids_movements)
state_all = np.array(iter_states[0])

for i in range(time_steps-1):
    state_all = np.vstack((state_all, iter_states[i+1]))

# compute fuel consumption for the optimal trajectory
def fuel_consumption(thrust, time_interval):
    return time_interval * sum(t.dot(t) for t in thrust)
print(f'Is fuel consumption {fuel_consumption(thrust_opt, time_interval)} lower than 250?')

# plt.figure()
# plot_state_trajectory(state_opt, universe)
fig, ax = plt.subplots()
plot_state_trajectory(state_all, universe)

# plot overall movement
fig, ax = plt.subplots()
plot_state_trajectory_movement(state_opt, asteroids_movements, universe)

# plot limits
plt.figure()
plot_rocket_limits(rocket, thrust_opt, state_opt)
