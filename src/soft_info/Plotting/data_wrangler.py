from typing import Dict, Tuple
import json
import numpy as np


def get_err_dicts(file_name) -> Tuple[Dict, Dict]:
    """ Returns two dictionaries:
        - err_per_job_dict: {method: {distance: {tot_shots: [], sum_errs: [], err_rates: [], sum_tot_shots: int, sum_err_rate: float, sum_sum_errs: int}}}
        - err_per_method_dict: {method: {d: [], errs: [], tot_shots: []}}
    """

    with open(file_name, 'r') as f:
        data = json.load(f)
    # print(f"Number of jobs: {len(data)}")

    err_per_job_dict = {}
    for idx, (job_id, job_content) in enumerate(data.items()):
        # if idx == 19:
        #     continue
        # if idx == 18:
        #     continue
        # print(f"Job ID: {job_id}")
        for distance, methods in job_content.get("distances", {}).items():
            for method, info in methods.items():
                if method == 'tot_shots_with_all_subsets':
                    tot_shots = int(info)
                    continue
                if method not in err_per_job_dict:
                    err_per_job_dict[method] = {}
                if int(distance) not in err_per_job_dict[method]:
                    err_per_job_dict[method][int(distance)] = { 'tot_shots': [], 'sum_errs': [], 'err_rates': []}
                err_per_job_dict[method][int(distance)]['tot_shots'].append(tot_shots)
                err_per_job_dict[method][int(distance)]['err_rates'].append(info['err_rate'])
                err_per_job_dict[method][int(distance)]['sum_errs'].append(info['sum_errs'])

    # Calculate mean error rates
    for method, distances in err_per_job_dict.items():
        for distance, data in distances.items():
            sum_tot_shots = sum(data['tot_shots'])
            sum_err_rate = sum(data['err_rates']) 
            sum_sum_errs = sum(data['sum_errs']) 

            err_per_job_dict[method][distance]['sum_tot_shots'] = sum_tot_shots
            err_per_job_dict[method][distance]['sum_err_rate'] = sum_err_rate
            err_per_job_dict[method][distance]['sum_sum_errs'] = sum_sum_errs

    err_per_method_dict = {}
    for method, distances in err_per_job_dict.items():
        err_per_method_dict[method] = {'d': [], 'errs': [], 'tot_shots': []}
        for distance, data in distances.items():
            err_per_method_dict[method]['d'].append(distance)
            err_per_method_dict[method]['errs'].append(data['sum_sum_errs'])
            err_per_method_dict[method]['tot_shots'].append(data['sum_tot_shots'])

    return err_per_job_dict, err_per_method_dict
    



def get_err_dicts_infoPerfo(file_name) -> Dict:
    """ Returns a dictionary:
        - err_list_per_method: {'nBits_list': [], 'method1': {d: ndarray}, 'method2': {d: ndarray}}
    """

    with open(file_name, 'r') as f:
        data = json.load(f)
    print(f"Number of jobs: {len(data)}\n")

    err_list_per_method = {}
    for job_id, job_data in data.items():

        nBits_list = job_data['additional_info']['nBits_list']
        if 'nBits_list' not in err_list_per_method:
            err_list_per_method['nBits_list'] = nBits_list
        else:
            assert err_list_per_method['nBits_list'] == nBits_list

        for distance, distance_data in job_data['distances'].items():
            for method, method_data in distance_data.items():

                if method == 'tot_shots_with_all_subsets':
                    continue
                if method[5:] not in err_list_per_method:
                    err_list_per_method[method[5:]] = {}
                if int(distance) not in err_list_per_method[method[5:]]:
                    err_list_per_method[method[5:]][int(distance)] = np.array(method_data['errs_per_bit'])
                else:
                    err_list_per_method[method[5:]][int(distance)] += np.array(method_data['errs_per_bit'])
                    # No need to divide by the number of jobs because I'll use the final value to divide anyways!

    return err_list_per_method

