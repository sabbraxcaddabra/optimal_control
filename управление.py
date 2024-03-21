import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

import matplotlib.pyplot as plt


H_LIST = np.array([0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000, 20000])
A_LIST = np.array([340.29, 340.10, 339.91, 339.53, 339.14, 338.76, 338.38, 337.98, 337.6, 337.21, 336.82, 336.43, 320.54, 299.53, 295.07])


MACH_LIST = np.array([0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                     1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                     2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6])

CX43_LIST = np.array([0.157, 0.158, 0.158, 0.160, 0.190, 0.325, 0.378, 0.385, 0.381, 0.371,
                    0.361, 0.351, 0.342, 0.332, 0.324, 0.316, 0.309, 0.303, 0.297,
                    0.292, 0.287, 0.283, 0.279, 0.277, 0.273, 0.270, 0.267, 0.265,
                    0.263, 0.263, 0.261, 0.260])

class ExtBalRS:

    def __init__(self, q, d, cx_list, mach_list, rho0=1.225, k=2, h=1e-3):

        self.q = q
        self.S = 0.25 * np.pi * d * d
        self.cx_list = cx_list
        self.mach_list = mach_list
        self.rho0 = rho0
        self.k = k
        self.h = h

    def cx(self, v, y):
        a_zv = np.interp(y, H_LIST, A_LIST)
        M = v / a_zv
        cx = np.interp(M, self.mach_list, self.cx_list)
        return cx

    def dcx_dv(self, v, y):
        dv = v * self.h
        return (self.cx(v+dv, y) - self.cx(v-dv, y)) / (2*dv)

    def dcx_dy(self, v, y):
        if y == 0:
            return 0
        dy = y * self.h
        return (self.cx(v, y+dy) - self.cx(v, y-dy)) / (2*dy)

    def H(self, y):
        return ((2e4 - y) / (2e4 + y))

    def dH_dy(self, y):
        return -(1/(y+2e4))-(2e4-y)/(y+2e4)**2


    def get_i(self, Pv, v, y):
        rho = self.rho0 * self.H(y)
        S = self.S
        q = self.q
        k = self.k
        return 0.5 * (Pv * S * k**2 * v**2 * self.cx(v, y) * rho)/q

    def external_bal_rs(self, t, y):
        p = y[4:]
        y = y[:4]
        Px, Py, Pv, Ptheta = p
        x, y, v, theta = y
        
        S = self.S
        q = self.q
        rho0 = self.rho0
        rho = rho0 * self.H(y)
        
        g = 9.80665
        # dy = np.zeros_like(y)
        # dp = np.zeros_like(p)

        # Расчет производных вектора состояния
        
        i = self.get_i(Pv, v, y)

        if i == 0:
            i = 1
        
        cx = self.cx(v, y)
        dx = v * np.cos(theta)  # Расчет приращения координаты X
        dy = v * np.sin(theta)  # Расчет приращения координаты Y
        Fx = -0.5 * i * cx * rho * v * v  # Расчет силы лобового сопротивления
        dv = (Fx * S - q * g * np.sin(theta)) / q  # Расчет приращения скорости V
        dtheta = -g * np.cos(theta) / v  # Расчет приращения угла teta

        # Расчет производных вектора сопряженных переменных

        dpv = -Pv * (0.5 * S * i * v**2 * self.dcx_dv(v, y) * rho - S * i * v * self.cx(v, y) * rho) / q
        dpv += Py * np.sin(theta)
        dpv += Ptheta * g * np.cos(theta) / v / v
        dpv += Px * np.cos(theta)

        dpx = 0

        dpy = -Pv * (0.5 * S * i * v**2 * self.cx(v, y) * self.dH_dy(y) * rho0 - 0.5 * S * i * v**2 * self.dcx_dy(v, y) * rho) / q

        dptheta = -(Px * v * np.sin(theta)) + Ptheta * g * np.sin(theta) / v + Py * v * np.cos(theta) - Pv * g * np.cos(theta)
        
        return [dx, dy, dv, dtheta, dpx, dpy, dpv, dptheta]


bal_rs = ExtBalRS(43, 0.1524, CX43_LIST, MACH_LIST)


v0 = 950
x0 = 0
y0 = 0
theta0 = np.deg2rad(45)
Px = Py = Pv = Ptheta = 0.0001


hit_ground = lambda t, y, *args: y[1]
hit_ground.terminal = True
hit_ground.direction = -1
state0 = [x0, y0, v0, theta0, 0, 0, 0, 0]

rho = 0.1
x_needed = 2e4

sol = solve_ivp(bal_rs.external_bal_rs,
                (0., 200),
                state0,
                dense_output=True,
                t_eval=np.linspace(0, 200, 200),
                events=hit_ground
                )

def f_nl(x, xf, rho):

    Px, Py, Pv, Ptheta = x
    state0 = [x0, y0, v0, theta0, Px, Py, Pv, Ptheta]


    sol = solve_ivp(bal_rs.external_bal_rs,
                    (0., 2000),
                    state0,
                    dense_output=True,
                    t_eval=np.linspace(0, 2000, 2000),
                    events=hit_ground
                   )
    events = sol.y_events

    x_last = events[0][0][0]


    return [0.5 * rho * (x_last - xf) ** 2] * 4

events = sol.y_events
sol = sol.sol

ts = np.linspace(sol.t_min, sol.t_max, 200)

x_opt, info, *_ = fsolve(f_nl, x0=np.array([Px, Py, Pv, Ptheta]), args=(x_needed, rho), xtol=1e-2, full_output=True)

print(f'Оптимальный вектор сопряженных переменных {x_opt=}')
print(info)


state_opt = [x0, y0, v0, theta0, *x_opt]

sol_opt = solve_ivp(bal_rs.external_bal_rs,
                (0., 200),
                state_opt,
                dense_output=True,
                t_eval=np.linspace(0, 200, 200),
                events=hit_ground
                )

sol_opt = sol_opt.sol
ts_opt = np.linspace(sol_opt.t_min, sol_opt.t_max, 200)
print(f'Отклонение от требуемой дальности {sol_opt(ts_opt)[0, -1] - x_needed:.0f}')

upr = bal_rs.get_i(sol_opt(ts_opt)[6], sol_opt(ts_opt)[2], sol_opt(ts_opt)[1])

fig, ax = plt.subplots()

ax.plot(sol(ts)[0], sol(ts)[1], label='Исходная траектория')
ax.plot(sol_opt(ts_opt)[0], sol_opt(ts_opt)[1], label='Траектория с управлением')
ax.set_xlabel('Дальность, м')
ax.set_ylabel('Высота, м')
ax.legend()
plt.show()

fig, ax = plt.subplots()

ax.plot(ts_opt, upr, label='Управление (Коэффициент ухудшения аэродинамики')
ax.set_xlabel('Время, с')
ax.set_ylabel('Управление, -')
ax.legend()
plt.show()






