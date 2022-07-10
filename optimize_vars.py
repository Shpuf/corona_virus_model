from scipy.optimize import differential_evolution, Bounds
from Virus import Virus, T_VACCINE, EVENT_ARR, CONTACT_TRACING_START
from data import Data
import numpy as np


def complete_cost(x, t_min, t_max):
    (beta_I, C_E, C_u, delta_R, delta_omega, min_omega, percent_asymptomatic, contact_trace_effect, kappa0, kappa1,
     kappa2, kappa3, kappa4, kappa5, kappa6, m0, m1, m2, m3, m4, m5, eta, vaccination_rate, vaccine_eligibility) = x

    data = Data()

    params = {
        'x': (beta_I, C_E, C_u, delta_R, delta_omega, min_omega),
        'kappa_array': [kappa0, kappa1, kappa2, kappa3, kappa4, kappa5, kappa6],
        'asymptomatic': percent_asymptomatic,
        'contact_trace': contact_trace_effect,
        'm_array': [m0, m1, m2, m3, m4, m5],
        'vaccine_parameters': (eta, vaccination_rate, vaccine_eligibility)
    }
    covid = make_virus(params=params)

    c_m, D, V = covid.results(t_min, t_max)

    try:
        m0 = c_m.size
        m1 = D.size
        m2 = V.size
        cost = (np.linalg.norm(data.reported_cases[int(t_min):int(t_max)] - c_m) / m0 +
                np.linalg.norm(data.reported_deaths[int(t_min):int(t_max)] - D) / m1 +
                np.linalg.norm(data.reported_vaccination[int(t_min):int(t_max)] - V) / m2)
        return cost

    except ValueError:
        return 99999.


def optimize(t_min, t_max, *, maxiter=15000, popsize=50, disp=False):
    bounds = Bounds(
        [
            0.00001,  # beta_I
            0.00001,  # C_E
            0.4,  # C_u
            7.,  # delta_R
            0.00001,  # delta_omega
            0.005,  # min_omega
            0.,  # contact_trace_effect
            # kappa 0-6:
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            # m 0-5:
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,  # eta
            0.,  # vaccination rate
            0.,  # vaccine eligibility
        ],
        [
            0.75,
            0.4,
            0.9,
            21.,
            0.2,
            0.022,
            0.4,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,

        ]
    )

    res = differential_evolution(
        complete_cost,
        bounds=bounds,
        args=(t_min, t_max),
        maxiter=maxiter,
        popsize=popsize,
        disp=disp
    )

    ret_dict = {
        'x': res.x[:6],
        'kappa_array': res.x[8:15],
        'asymptomatic': res.x[6],
        'contact_trace': res.x[7],
        'm_array': res.x[15:21],
        'vaccine_parameters': res.x[21:]
    }

    return ret_dict


def transform_array(arr):
    m = len(arr)//2
    for i in range(len(arr)):
        arr[i] = arr[i] * (i - m) ** 2
    return arr


def cost(x, t_min, t_max, params: dict = None, *, print_cost=False):
    """
    should only be used with optimize() func
    :param x:
    :param t_min:
    :param t_max:
    :param params:
    :param print_cost:
    :return:
    """
    data = Data()
    if x is None:
        make_virus(params)


    if params is None:
        # optimize the basic virus parameters
        beta_I, C_E, C_u, delta_R, delta_omega, min_omega, kappa0, contact_trace_effect = x
        kappa_array = [kappa0]
        vaccine_parameters = None
        m_array = None

    else:
        assert 'x' in params, "x field in params must be defined in order to keep optimizing"
        assert 'y0' in params, "y0 must be defined in order to keep optimizing"
        beta_I, C_E, C_u, delta_R, delta_omega, min_omega = params['x']
        vaccine_parameters = assign_if_in_dict_else_None('vaccine_parameters', params)
        contact_trace_effect = assign_if_in_dict_else_None('contact_trace', params)
        kappa_array = assign_if_in_dict_else_None('kappa_array', params)
        m_array = assign_if_in_dict_else_None('m_array', params)
        if kappa_array is not None:
            kappa_array = list(kappa_array)
        if m_array is not None:
            m_array = list(m_array)
        # optimize only the condition parameters

        if t_min == T_VACCINE:
            assert x.size == 5, "with t == T_VACCINE must optimize for vaccine variables"
            curr_kappa, curr_m, eta, vaccination_rate, vaccine_eligibility = x
            vaccine_parameters = np.array([eta, vaccination_rate, vaccine_eligibility])
        else:
            curr_kappa, curr_m = x
        kappa_array.append(curr_kappa)
        if m_array is not None:
            m_array.append(curr_m)
        else:
            m_array = [curr_m]
    virus_parameters = [beta_I, C_E, C_u, delta_R, delta_omega, min_omega]
    covid = Virus(
        virus_params=virus_parameters,
        vaccine_params=vaccine_parameters,
        contact_trace_effect=contact_trace_effect,
        kappa_array=kappa_array,
        m_array=m_array
    )
    c_m, D, V = covid.results(t_min, t_max, y0=params['y0'] if params is not None else None)
    try:
        m0 = c_m.size
        m1 = D.size
        m2 = V.size
        cost = (np.linalg.norm(data.reported_cases[int(t_min):int(t_max)] - c_m) / m0 +
                np.linalg.norm(data.reported_deaths[int(t_min):int(t_max)] - D) / m1 +
                np.linalg.norm(data.reported_vaccination[int(t_min):int(t_max)] - V) / m2)
        if print_cost:
            print(cost)
        return cost

    except ValueError:
        return 99999.


def differential_evolution_wrapper(t_min, t_max, *, params: dict = None, first_iter=False, optimize_vaccine=False,
                                   maxiter=15000, popsize=50, disp=False):
    assert not (first_iter and optimize_vaccine), "shouldn't optimize vaccine vars at first iter"
    if params is None:
        assert first_iter, "params must be included beyond first iteration."
    else:
        assert 'x' in params and len(params['x']) == 6, ("x field in parameter dictionary must be occupied and "
                                                         "containing the basic virus variables "
                                                         "(beta_I, C_E, C_u, delta_R, delta_omega, min_omega)"
                                                         "after first iteration.")
        if t_min > T_VACCINE:
            assert 'vaccine_parameters' in params
    if first_iter:
        bounds = Bounds(
            [
                0.00001,  # beta_I
                0.00001,  # C_E
                0.4,  # C_u
                7.,  # delta_R
                0.00001,  # delta_omega
                0.005,  # min_omega
                0.0001,  # kappa0
                0.0001  # contact_trace_effect
            ],
            [
                0.75,
                0.4,
                0.9,
                21.,
                0.2,
                0.022,
                0.2,
                0.4
            ]
        )
    elif optimize_vaccine:
        bounds = Bounds(
            [
                0.0001,  # kappa
                0.0001,  # m
                0.0001,  # eta
                0.0001,  # vaccination rate
                0.0001  # vaccine eligibility
            ],
            [
                0.2,
                1.,
                1.,
                1.,
                1.
            ]
        )
    else:
        bounds = Bounds(
            [
                0.0001,  # kappa
                0.0001  # m
            ],
            [
                0.2,
                1.
            ]
        )
    res = differential_evolution(
        cost,
        args=(
            t_min,
            t_max,
            params
        ),
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        disp=disp
    )
    if first_iter:
        params = {'x': res.x[:6], 'kappa_array': [res.x[6]], 'contact_trace': res.x[7]}
    elif optimize_vaccine:
        params['vaccine_parameters'] = res.x[2:]
        params['kappa_array'].append(res.x[0])
        params['m_array'].append(res.x[1])
    else:
        if 'm_array' not in params:
            params['m_array'] = list()
        params['kappa_array'].append(res.x[0])
        params['m_array'].append(res.x[1])
    return params


def optimize_by_parts(first_stage_only=False, skip_first_stage=False, params=None, *, disp=False):
    lambda_array = EVENT_ARR

    if not skip_first_stage:
        # setting up the basic virus params
        params = differential_evolution_wrapper(t_min=0, t_max=lambda_array[1], first_iter=True, disp=disp)
        params['y0'] = make_virus(params).results(t_min=0, t_max=lambda_array[1], get_y=True)
    else:
        assert params is not None, "to skip first stage of optimizing params must be initialized"

    if not first_stage_only:
        for t_min, t_max in zip(lambda_array[1:-1], lambda_array[2:]):
            if t_min == T_VACCINE:
                params = differential_evolution_wrapper(t_min=t_min, t_max=t_max, params=params,
                                                        optimize_vaccine=True, disp=disp)
            else:
                params = differential_evolution_wrapper(t_min=t_min, t_max=t_max, params=params, disp=disp)
            params['y0'] = make_virus(params).results(t_min=t_min, t_max=t_max, y0=params['y0'], get_y=True)
    return params


def assign_if_in_dict_else_None(kw: str, dic: dict):
    if kw in dic:
        return dic[kw]
    else:
        return None


def make_virus(params: dict) -> Virus:
    if isinstance(params['x'], np.ndarray):
        virus_params = params['x'].tolist()
    else:
        virus_params = params['x']
    vaccine_params = assign_if_in_dict_else_None('vaccine_parameters', params)
    contact_trace = assign_if_in_dict_else_None('contact_trace', params)
    kappa_array = assign_if_in_dict_else_None('kappa_array', params)
    m_array = assign_if_in_dict_else_None('m_array', params)
    return Virus(
        virus_params=virus_params,
        vaccine_params=vaccine_params,
        contact_trace_effect=contact_trace,
        kappa_array=kappa_array,
        m_array=m_array
    )
