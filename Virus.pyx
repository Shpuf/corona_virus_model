#cython: language_level=3

import numpy as np
cimport numpy as np
from scipy.integrate import solve_ivp

np.import_array()
np.set_printoptions(suppress=True)
DTYPE = np.float
ctypedef np.float_t DTYPE_t
# using constants so implementing the model for other countries would be easier

T_VACCINE = 332.  # vaccine rollout t

EVENT_ARR = np.array([22., 58., 210., 260., 310., 332., 352.], dtype=float)  # t of government actions,
# should include T_VACCINE

T_END = 400. # data includes data up to day 596, doing till dat 450 to save calculation time

CONTACT_TRACING_START = 24.  # t of contact tracing start, afterward the amount of un-identified carriers of the
# disease is lowered dramatically

GLOBAL_N = 8000000.  # population of chosen country

CASES_AND_DEATHS_FILE_PATH = r"./COVID-19-cases-Israel.txt"
VACCINATION_FILE_PATH = r"./Vaccination-Data-Israel.txt"
# ^ self explained

class Virus:
    lambda_array : np.ndarray = EVENT_ARR
    d_E : float = 5.5
    gamma_E : float = 1 / d_E
    d_I : float = 6.7
    d_g : float = 6.
    d_I_u : float = 7.3
    contact_tracing_start : float = CONTACT_TRACING_START
    N : float = GLOBAL_N
    vaccine_rollout_date : float = T_VACCINE
    percent_asymptomatic : float = 0.6

    def __init__(
            self,
            *,
            virus_params,
            vaccine_params = None,
            contact_trace_effect = None,
            kappa_array = None,
            m_array = None
    ):
        self.beta_I, self.C_E, self.C_u, self.delta_R, self.delta_omega, self.min_omega = virus_params
        if vaccine_params is not None:
            if vaccine_params.size == 3:
                self.eta, self.vaccination_rate, self.vaccine_eligibility = vaccine_params
                self.mu = self.eta
            elif vaccine_params.size == 4:
                self.eta, self.vaccination_rate, self.vaccine_eligibility, self.mu = vaccine_params
            else:
                raise ValueError, "vaccine_params must be of size 3 or 4 or be none."
        else:
            # assign meaningless non zero value so won't be necessary to include when irrelevant
            self.eta = 0.5
            self.vaccination_rate = 0.5
            self.vaccine_eligibility = 0.5
        if contact_trace_effect is not None:
            self.contact_trace_effect = contact_trace_effect
        else:
            contact_trace_effect = 0.4  # same as vaccine params, meaningless value, done so it wont have to be included
                                        # when optimizing for time ranges when it doesnt matter,
                                        # non zero so there won't be any accidental divide by zero errors.
        self.beta_I_min = self.C_u * self.beta_I
        self.beta_E = self.C_E * self.beta_I
        self.max_omega = self.min_omega + self.delta_omega

        if kappa_array is not None:
            self.kappa_array = list(kappa_array)
            if len(self.kappa_array) != len(self.lambda_array):
                self.kappa_array += [0 for i in range((len(self.lambda_array) - len(self.kappa_array)))]
                self.kappa_array = np.array(self.kappa_array, dtype=float)
        else:
            self.kappa_array = np.array([0 for i in range(len(self.lambda_array))], dtype=float)
        if m_array is not None:
            self.m_array = list(m_array)
            if len(self.m_array) != len(self.lambda_array):
                self.m_array += [0 for i in range((len(self.lambda_array) - len(self.m_array)))]
            self.m_array = np.array(self.m_array, dtype=float)
        else:
            self.m_array=np.array([0.5 for i in range(len(self.lambda_array))], dtype=float)

    def alpha(self, float t) -> float:
        """
        :param t: current day of the pandemic
        :return: either 0 if t <= the date in which vaccines become available else returns the vaccination rate
        """
        return (t >= self.vaccine_rollout_date) * self.vaccination_rate

    def theta(self, float t) -> float:
        """
        a function describing the dynamical nature of the proportion of asymptomatic infected
        :param t: current day of the pandemic
        :return: proportion of people who receive the virus symptomatically
        """
        if t < self.contact_tracing_start:
            return self.percent_asymptomatic
        elif self.contact_tracing_start <= t < self.contact_tracing_start + 15:
            return self.percent_asymptomatic + (t - self.contact_tracing_start) * \
                   self.contact_trace_effect / 15
        else:
            return self.percent_asymptomatic + self.contact_trace_effect

    def m(self, float t) -> float:
        """
        a function describing the current strength of government mitigation actions of the virus
        :param t: current day of the pandemic
        :return: current mitigation strength
        """
        if np.less_equal(t, self.lambda_array[0]):
            return 1.
        if t <= self.lambda_array[-1]:
            i = np.where(np.logical_and(np.less_equal(t, self.lambda_array[1:]),
                                        np.greater(t, self.lambda_array[:-1])))[0][0]
        else:
            i = self.lambda_array.size - 1
        if i == 0:
            return np.exp(-self.kappa_array[i] * (t - self.lambda_array[i]))
        return (self.m(self.lambda_array[i]) - self.m_array[i - 1]) * (
            np.exp(-self.kappa_array[i] * (t - self.lambda_array[i]))) + self.m_array[i - 1]

    m_E = m_I_u = m_H_D = m_H_R = m_I = m

    def g(self, float t) -> float:
        return self.d_g * (1 - self.m_I(t))

    def gamma_I(self, float t) -> float:
        return 1 / (self.d_I - self.g(t))

    def gamma_I_u(self, float t) -> float:
        return 1 / (self.d_I_u + self.g(t))

    def gamma_H_R(self, float t) -> float:
        return 1 / (self.d_I_u + self.g(t))

    def gamma_H_D(self, float t) -> float:
        return 1 / (self.d_I_u + self.g(t) + self.delta_R)

    def omega(self, float t) -> float:
        return self.m_I(t) * self.max_omega + (1 - self.m_I(t)) * self.min_omega

    def beta_I_u(self, float t) -> float:
        return self.beta_I_min + (self.beta_I - self.beta_I_min) / (1 - self.omega(t) * (1 - self.theta(t)))

    def C_H(self, float t) -> float:
        return 0.0275 * ((self.beta_I / self.gamma_I(t)) + (self.beta_E / self.gamma_E) + (1 - self.theta(t)) *
            (self.beta_I_u(t) / self.gamma_I_u(t))) / ((1 - 0.0275) * self.beta_I * self.theta(t) *
                ((1 - self.omega(t) / self.theta(t)) * (1 / self.gamma_H_R(t)) +
                     (self.omega(t) / self.theta(t)) * (1 / self.gamma_H_D(t))))

    def beta_H_R(self, t) -> float:
        return self.C_H(t) * self.beta_I

    def beta_H_D(self, t) -> float:
        return self.C_H(t) * self.beta_I

    def derivatives(self, float t, np.ndarray y) -> np.ndarray:
        S, E, I, I_u, H_D, H_R, R_d, R_u, D, V = y
        dS_dt = -(S / self.N) * (self.m_E(t) * self.beta_E * E + self.m_I(t) * self.beta_I * I + self.m_I_u(
                t) * self.beta_I_u(t) * I_u) - (S / self.N) * (self.m_H_R(t) * self.beta_H_R(
                    t) * H_R + self.m_H_D(t) * self.beta_H_D(t) * H_D) - (1 - V / (
                        self.N * self.vaccine_eligibility)) * self.alpha(t) * self.eta * S
        dE_dt = S / self.N * (self.m_E(t) * self.beta_E * E + self.m_I(t) * self.beta_I * I + self.m_I_u(
                t) * self.beta_I_u(t)) + S / self.N * (
                self.m_H_R(t) * self.beta_H_R(t) * H_R + self.m_H_D(t) * self.beta_H_D(
                        t) * H_D) - self.gamma_E * E
        dI_dt = self.gamma_E * E - self.gamma_I(t) * I
        dI_udt = (1 - self.theta(t)) * self.gamma_I(t) * I - self.gamma_I_u(t) * I_u
        dH_Rdt = self.theta(t) * (1 - self.omega(t) / self.theta(t)) * self.gamma_I(t) * I - self.gamma_H_R(t) * H_R
        dH_Ddt = self.omega(t) * self.gamma_I(t) * I - self.gamma_H_D(t) * H_D
        dR_ddt = self.gamma_H_R(t) * H_R
        dR_udt = self.gamma_I_u(t) * I_u - (1 - V / (
                        self.N * self.vaccine_eligibility)) * self.alpha(t) * self.eta * R_u
        dD_dt = self.gamma_H_D(t) * H_D
        dV_dt = (1 - V / (self.N * self.vaccine_eligibility + 1)) * self.alpha(t) * self.eta * (S + R_u)
        return np.array([dS_dt, dE_dt, dI_dt, dI_udt, dH_Rdt, dH_Ddt, dR_ddt, dR_udt, dD_dt, dV_dt])

    def results(self,
                float t_min,
                float t_max,
                np.ndarray y0 = None,
                *,
                get_y=False
    ):
        N = GLOBAL_N
        if y0 is None:
            y0 = np.array([N, 0., 0., 1., 0., 0., 0., 0., 0., 0.], dtype=float)

        sol = solve_ivp(
            self.derivatives,
            (t_min, t_max),
            y0=y0,
            t_eval=np.arange(t_min, t_max),
            vectorized=True,
            method="LSODA",

        )

        S, E, I, I_u, H_R, H_D, R_d, R_u, D, V = sol.y
        c_m = H_R + H_D + R_d + D
        if not get_y:
            return c_m, D, V
        y = sol.y[:, -1]
        return y


if __name__ == '__main__':
    pass