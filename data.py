from Virus import CASES_AND_DEATHS_FILE_PATH, VACCINATION_FILE_PATH, T_VACCINE
import numpy as np


class Data:
    """
    import covid data and vaccination data from ministry of health
    """

    _file_name = CASES_AND_DEATHS_FILE_PATH
    _covid_data = np.genfromtxt(_file_name)
    reported_cases = _covid_data[:, 3]
    reported_deaths = _covid_data[:, 1]

    _file_name = VACCINATION_FILE_PATH
    _vaccination_data = np.genfromtxt(_file_name, skip_header=8)
    _v_r = _vaccination_data[:, 2]
    _zeros = np.zeros(int(T_VACCINE))
    reported_vaccination = np.concatenate((_zeros, _v_r), axis=0)