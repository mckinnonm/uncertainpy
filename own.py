import uncertainpy
import numpy as np
import odespy

class YourOwnCoffeeModel(uncertainpy.Model):
    def __init__(self, parameters=None):
        uncertainpy.Model.__init__(self, parameters=parameters)

        self.kappa = -0.01
        self.u_env = 20

        self.u0 = 95
        self.t_points = np.linspace(0, 360, 150)
        self.x = np.linspace(0, 10, 500)


    def f(self, u, t):
        return self.kappa*(u - self.u_env)

    def run(self):
        solver = odespy.RK4(self.f)
        solver.set_initial_condition(self.u0)

        self.U, self.t = solver.solve(self.t_points)
        return self.t, self.U