from result_saver import SaverProvider
from Scratch import metadata_loader, find_closest_calib_jobs, load_calibration_memory

from soft_info import (get_noise_dict_from_backend, get_avgs_from_dict,
                       get_repcode_IQ_map, RepetitionCodeStimCircuit,
                       inv_qubit_mapping, gaussianIQConvertor, get_cols_to_keep,
                       generate_subsets_with_center, get_subsample_layout,
                       plot_IQ_data_with_countMat, RepCodeIQSimulator)

from src import cpp_soft_info as csi

import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep
from datetime import datetime
import json
import os



def decoder(provider, DEVICE, LOGICAL, XBASIS, ROUNDS, file_name, threshold=0.01):
    md = load_metadata(DEVICE, LOGICAL, XBASIS, ROUNDS)
    jobs_by_calib_date = organize_jobs_by_calibration_date(md)

    idx = 0
    for calib_date, job_ids in jobs_by_calib_date.items():
        idx += 1
        if idx > 2:
            break
        kde_dict, kde_dict_PS, msmt_error_dict, noise_dict = retrieve_calib_data(provider, DEVICE, calib_date)
        big_memory, d, T, link_qubits, code_qubits, shots_per_job = retrieve_memories_and_job_info(provider, jobs_by_calib_date[calib_date], md)
        inverted_q_map = inv_qubit_mapping(get_repcode_IQ_map(link_qubits+code_qubits, T))

        # pSoft, ratio, countMat, pSoft_PS, ratio_PS, countMat_PS = get_pSoft_and_countMat(big_memory, kde_dict, kde_dict_PS, inverted_q_map, threshold, plot = True)
        # pSoft_PS, ratio_PS, countMat_PS = get_pSoft_and_countMat(big_memory, kde_dict, kde_dict_PS, inverted_q_map, threshold, plot = True)
        pSoft_PS, ratio_PS, countMat_PS = get_pSoft_and_countMat(big_memory, kde_dict, kde_dict_PS, inverted_q_map, threshold, plot = False)
        # pSoft_mean = np.mean(pSoft)
        pSoft_PS_mean = np.mean(pSoft_PS)

        distances = np.arange(3, d+1, 2)[::-1]
        for d_small in tqdm(distances, desc=f"distance"):
            # pSoft_subset_big, countMat_subset_big, num_subsets, num_shots_subset_tot = get_big_subset_mats(d_small, T, d, pSoft, countMat)
            pSoft_subset_big_PS, countMat_subset_big_PS, num_subsets_PS, num_shots_PS_subset_tot = get_big_subset_mats(d_small, T, d, pSoft_PS, countMat_PS)

            # assert num_shots_subset_tot == num_shots_PS_subset_tot, "num_shots and num_shots_PS should be equal"
            
            _RESETS = False
            # model_mean_for_soft, model_mean_for_hard, model_mean_mean_for_hard, model_mean_mean_for_hard_PS, noise_list = get_stim_models(noise_dict, msmt_error_dict, pSoft_mean, pSoft_PS_mean, d_small,
            #                                                                                                           d, T, LOGICAL, XBASIS, _RESETS, link_qubits, code_qubits)
            model_mean_for_soft, model_mean_for_hard, model_mean_mean_for_hard, model_mean_mean_for_hard_PS, noise_list = get_stim_models(noise_dict, msmt_error_dict, 0.4, pSoft_PS_mean, d_small,
                                                                                                                      d, T, LOGICAL, XBASIS, _RESETS, link_qubits, code_qubits)

            # res_s_K = csi.decodeConvertorAll(model_mean_for_soft, countMat_subset_big, pSoft_subset_big, T, 
            #                                     int(LOGICAL), _RESETS, decode_hard=False)
            # res_h_K = csi.decodeConvertorAll(model_mean_for_hard, countMat_subset_big, pSoft_subset_big, T,
            #                                                 int(LOGICAL), _RESETS, decode_hard=True)
            # res_h_K_mean = csi.decodeConvertorAll(model_mean_mean_for_hard, countMat_subset_big, pSoft_subset_big, T,
            #                                                 int(LOGICAL), _RESETS, decode_hard=True)

            res_s_KPS = csi.decodeConvertorAll(model_mean_for_soft, countMat_subset_big_PS, pSoft_subset_big_PS, T,
                                                            int(LOGICAL), _RESETS, decode_hard=False)
            # res_h_KPS = csi.decodeConvertorAll(model_mean_for_hard, countMat_subset_big_PS, pSoft_subset_big_PS, T,
            #                                                 int(LOGICAL), _RESETS, decode_hard=True)    
            # res_h_K_meanPS = csi.decodeConvertorAll(model_mean_mean_for_hard_PS, countMat_subset_big_PS, pSoft_subset_big_PS, T,
            #                                                 int(LOGICAL), _RESETS, decode_hard=True)
            

            result_dict = {
                # 'dict_s_K': res_to_job_subset_res(res_s_K, shots_per_job, num_subsets, job_ids),
                # 'dict_h_K': res_to_job_subset_res(res_h_K, shots_per_job, num_subsets, job_ids),
                # 'dict_h_K_mean': res_to_job_subset_res(res_h_K_mean, shots_per_job, num_subsets, job_ids),
                # 'dict_s_KPS': res_to_job_subset_res(res_s_KPS, shots_per_job, num_subsets_PS, job_ids),
                'dict_s_KPS_no_ambig': res_to_job_subset_res(res_s_KPS, shots_per_job, num_subsets_PS, job_ids),
                # 'dict_h_KPS': res_to_job_subset_res(res_h_KPS, shots_per_job, num_subsets_PS, job_ids),
                # 'dict_h_K_meanPS': res_to_job_subset_res(res_h_K_meanPS, shots_per_job, num_subsets_PS, job_ids),
            }

            if not os.path.exists(file_name):
                data = {}
            else:
                with open(file_name, "r") as f:
                    data = json.load(f)

            for job_id in job_ids:
                if job_id not in data:
                    data[job_id] = {"additional_info": {
                        'threshold': threshold,
                        'ratio_leaked_PS': ratio_PS,
                        'pSoft_PS_mean': pSoft_PS_mean,
                        'noise_list': [float(noise) for noise in noise_list],
                    }, "distances": {}}

                if str(d_small) not in data[job_id]['distances']:
                    data[job_id]["distances"][str(d_small)] = {'tot_shots_with_all_subsets': float(num_shots_PS_subset_tot/len(job_ids))}

                for method, res_dict in result_dict.items():
                    data[job_id]['distances'][str(d_small)][method[5:]] = {
                        'err_rate': np.sum(res_dict[job_id])/ (num_shots_PS_subset_tot/len(job_ids)),
                        'sum_errs': float(np.sum(res_dict[job_id])),
                        'errs': res_dict[job_id],
                    }

            with open(file_name, "w") as f:
                json.dump(data, f, indent=4)








def decoderSimul(provider, DEVICE, LOGICAL, XBASIS, ROUNDS, file_name, threshold=0, shots_mult = 20, multiplier=1.0):

    calib_date = '2024-03-24'
 
    assert DEVICE == 'ibm_sherbrooke', "Only IBM Sherbrooke is supported for now" # for _is_hex

    d = 52 # as device
    T = ROUNDS
    simulator = RepCodeIQSimulator(provider, d=d, T=ROUNDS, device=DEVICE,
                                   _is_hex=True, other_date=calib_date, double_msmt=True, multiplier=multiplier)
    
    link_qubits = simulator.layout[:d]
    code_qubits = simulator.layout[d:]
    inverted_q_map = inv_qubit_mapping(get_repcode_IQ_map(link_qubits+code_qubits, T))
    job_ids = ['simulator_job_60_high_d']

    shots = int(4e6/((d-1)*T+2*d-1)) * shots_mult
    big_memory = simulator.generate_IQ(shots, logical='0', xbasis=XBASIS, resets=False, individual=False)

    kde_dict_PS = csi.get_KDEs(simulator.all_memories, [0.6], relError=1, absError=-1, num_points=20)
    msmt_error_dict = simulator.msmt_err_dict
    noise_dict = simulator.noise_dict


    pSoft_PS, ratio_PS, countMat_PS = get_pSoft_and_countMat(big_memory, None, kde_dict_PS, inverted_q_map, threshold, plot = False)
    pSoft_PS_mean = np.mean(pSoft_PS)

    distances = np.arange(3, d+1, 2)[::-1]
    # distances = np.arange(29, d+1, 2)[::-1]
    for d_small in tqdm(distances, desc=f"distance"):
        pSoft_subset_big_PS, countMat_subset_big_PS, num_subsets_PS, num_shots_PS_subset_tot = get_big_subset_mats(d_small, T, d, pSoft_PS, countMat_PS)

        _RESETS = False
        model_mean_for_soft, model_mean_for_hard, model_mean_mean_for_hard, model_mean_mean_for_hard_PS, noise_list = get_stim_models(noise_dict, msmt_error_dict, 0.4, pSoft_PS_mean, d_small,
                                                                                                                    d, T, LOGICAL, XBASIS, _RESETS, link_qubits, code_qubits, noise_list=simulator.noise_list)

        print(f"\nnoise_list simulator: {noise_list}")
        res_s_KPS = csi.decodeConvertorAll(model_mean_for_soft, countMat_subset_big_PS, pSoft_subset_big_PS, T,
                                                        int(LOGICAL), _RESETS, decode_hard=False)
        res_h_KPS = csi.decodeConvertorAll(model_mean_for_hard, countMat_subset_big_PS, pSoft_subset_big_PS, T,
                                                        int(LOGICAL), _RESETS, decode_hard=True)    
        

        result_dict = {
            'dict_s_KPS': res_to_job_subset_res(res_s_KPS, shots, num_subsets_PS, job_ids),
            'dict_h_KPS': res_to_job_subset_res(res_h_KPS, shots, num_subsets_PS, job_ids),
        }

        if not os.path.exists(file_name):
            data = {}
        else:
            with open(file_name, "r") as f:
                data = json.load(f)

        for job_id in job_ids:
            if job_id not in data:
                data[job_id] = {"additional_info": {
                    'threshold': threshold,
                    'ratio_leaked_PS': ratio_PS,
                    'pSoft_PS_mean': pSoft_PS_mean,
                    'noise_list': [float(noise) for noise in noise_list],
                }, "distances": {}}

            if str(d_small) not in data[job_id]['distances']:
                data[job_id]["distances"][str(d_small)] = {'tot_shots_with_all_subsets': float(num_shots_PS_subset_tot/len(job_ids))}

            for method, res_dict in result_dict.items():
                data[job_id]['distances'][str(d_small)][method[5:]] = {
                    'err_rate': np.sum(res_dict[job_id])/ (num_shots_PS_subset_tot/len(job_ids)),
                    'sum_errs': float(np.sum(res_dict[job_id])),
                    'errs': res_dict[job_id],
                }

        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)




def decoderSimulInfoPerfo(provider, DEVICE, LOGICAL, XBASIS, ROUNDS, file_name, threshold=0, shots_mult = 20, multiplier=1.0, distances=None):

    calib_date = '2024-03-24'
 
    assert DEVICE == 'ibm_sherbrooke', "Only IBM Sherbrooke is supported for now" # for _is_hex

    d = 52 # as device
    T = ROUNDS
    simulator = RepCodeIQSimulator(provider, d=d, T=ROUNDS, device=DEVICE,
                                   _is_hex=True, other_date=calib_date, double_msmt=True, multiplier=multiplier)
    
    link_qubits = simulator.layout[:d]
    code_qubits = simulator.layout[d:]
    inverted_q_map = inv_qubit_mapping(get_repcode_IQ_map(link_qubits+code_qubits, T))
    job_ids = ['simulator_job'+str(multiplier)+str(shots_mult)]

    shots = int(4e6/((d-1)*T+2*d-1)) * shots_mult
    big_memory = simulator.generate_IQ(shots, logical='0', xbasis=XBASIS, resets=False, individual=False)

    kde_dict_PS = csi.get_KDEs(simulator.all_memories, [0.6], relError=1, absError=-1, num_points=20)
    msmt_error_dict = simulator.msmt_err_dict
    noise_dict = simulator.noise_dict


    pSoft_PS, ratio_PS, countMat_PS = get_pSoft_and_countMat(big_memory, None, kde_dict_PS, inverted_q_map, threshold, plot = False)
    pSoft_PS_mean = np.mean(pSoft_PS)

    distances = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 27, 33, 39] if distances is None else distances
    print(distances)
    # distances = [17, 19, 21, 27, 33, 39] 
    for d_small in tqdm(distances, desc=f"distance"):
        pSoft_subset_big_PS, countMat_subset_big_PS, num_subsets_PS, num_shots_PS_subset_tot = get_big_subset_mats(d_small, T, d, pSoft_PS, countMat_PS)

        _RESETS = False
        model_mean_for_soft, model_mean_for_hard, model_mean_mean_for_hard, model_mean_mean_for_hard_PS, noise_list = get_stim_models(noise_dict, msmt_error_dict, 0.4, pSoft_PS_mean, d_small,
                                                                                                                    d, T, LOGICAL, XBASIS, _RESETS, link_qubits, code_qubits, noise_list=simulator.noise_list)

        print(f"\nnoise_list simulator: {noise_list}")


        nBits_res_dict = {}
        nBits_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, -1]
        for nBits in nBits_range:
            if nBits == -1:
                pSoft_subset_big_PS_quantized = pSoft_subset_big_PS
            else:
                pSoft_subset_big_PS_quantized = csi.quantizeMatrixVectorized(pSoft_subset_big_PS, nBits)

            res_s_KPS = csi.decodeConvertorAll(model_mean_for_soft, countMat_subset_big_PS, pSoft_subset_big_PS_quantized, T,
                                                            int(LOGICAL), _RESETS, decode_hard=False)

            result_dict = {
                'dict_s_KPS': res_to_job_subset_res(res_s_KPS, shots, num_subsets_PS, job_ids),
                # 'dict_h_KPS': res_to_job_subset_res(res_h_KPS, shots, num_subsets_PS, job_ids),
            }
            for method, job_ids_dict in result_dict.items():
                for job_id, err_per_subset_list in job_ids_dict.items():
                    if method not in nBits_res_dict:
                        nBits_res_dict[method] = {}
                    if job_id not in nBits_res_dict[method]:
                        nBits_res_dict[method][job_id] = []
                    nBits_res_dict[method][job_id].append(sum(err_per_subset_list))


        if not os.path.exists(file_name):
            data = {}
        else:
            with open(file_name, "r") as f:
                data = json.load(f)

        for job_id in job_ids:
            if job_id not in data:
                data[job_id] = {"additional_info": {
                    'threshold': threshold,
                    'ratio_leaked_PS': ratio_PS,
                    'pSoft_PS_mean': pSoft_PS_mean,
                    'noise_list': [float(noise) for noise in noise_list],
                    'nBits_list': nBits_range,
                }, "distances": {}}

            if str(d_small) not in data[job_id]['distances']:
                data[job_id]["distances"][str(d_small)] = {'tot_shots_with_all_subsets': float(num_shots_PS_subset_tot/len(job_ids))}

            for method, res_dict in nBits_res_dict.items():
                data[job_id]['distances'][str(d_small)][method] = {
                    'errs_per_bit': res_dict[job_id]
                }

        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)






def decoderGaussian(provider, DEVICE, LOGICAL, XBASIS, ROUNDS, file_name, threshold=0.01):
    md = load_metadata(DEVICE, LOGICAL, XBASIS, ROUNDS)
    jobs_by_calib_date = organize_jobs_by_calibration_date(md)

    idx = 0
    for calib_date, job_ids in jobs_by_calib_date.items():
        idx += 1
        if idx > 2:
            break
        kde_dict, kde_dict_PS, msmt_error_dict, noise_dict, gmm_dict = retrieve_calib_data(provider, DEVICE, calib_date, return_gmm=True)
        big_memory, d, T, link_qubits, code_qubits, shots_per_job = retrieve_memories_and_job_info(provider, jobs_by_calib_date[calib_date], md)
        inverted_q_map = inv_qubit_mapping(get_repcode_IQ_map(link_qubits+code_qubits, T))

        print(f"\nStarting to get pSoft at {datetime.now()}")
        countMatG, pSoftG = gaussianIQConvertor(big_memory, inverted_q_map, gmm_dict)
        pSoftG_mean = np.mean(pSoftG)
        print(f"Finished getting pSofts at {datetime.now()}\n")

        distances = np.arange(3, d+1, 2)[::-1]
        for d_small in tqdm(distances, desc=f"distance"):
            pSoftG_subset_big, countMatG_subset_big, num_subsets, num_shots_subset_tot = get_big_subset_mats(d_small, T, d, pSoftG, countMatG)

            
            _RESETS = False
            model_mean_for_soft, model_mean_for_hard, model_mean_mean_for_hard, model_mean_mean_for_hard_PS, noise_list = get_stim_models(noise_dict, msmt_error_dict, 0.4, pSoftG_mean , d_small,
                                                                                                                      d, T, LOGICAL, XBASIS, _RESETS, link_qubits, code_qubits)

            res_s_G = csi.decodeConvertorAll(model_mean_for_soft, countMatG_subset_big, pSoftG_subset_big, T,
                                                            int(LOGICAL), _RESETS, decode_hard=False)
            res_h_G = csi.decodeConvertorAll(model_mean_for_hard, countMatG_subset_big, pSoftG_subset_big, T,
                                                            int(LOGICAL), _RESETS, decode_hard=True)
            

            result_dict = {
                'dict_s_G': res_to_job_subset_res(res_s_G, shots_per_job, num_subsets, job_ids),   
                'dict_h_G': res_to_job_subset_res(res_h_G, shots_per_job, num_subsets, job_ids),
            }

            if not os.path.exists(file_name):
                data = {}
            else:
                with open(file_name, "r") as f:
                    data = json.load(f)

            for job_id in job_ids:
                if job_id not in data:
                    data[job_id] = {"additional_info": {
                        'threshold': threshold,
                        'noise_list': [float(noise) for noise in noise_list],
                    }, "distances": {}}

                if str(d_small) not in data[job_id]['distances']:
                    data[job_id]["distances"][str(d_small)] = {'tot_shots_with_all_subsets': float(num_shots_subset_tot/len(job_ids))}

                for method, res_dict in result_dict.items():
                    data[job_id]['distances'][str(d_small)][method[5:]] = {
                        'err_rate': np.sum(res_dict[job_id])/ (num_shots_subset_tot/len(job_ids)),
                        'sum_errs': float(np.sum(res_dict[job_id])),
                        'errs': res_dict[job_id],
                    }

            with open(file_name, "w") as f:
                json.dump(data, f, indent=4)












def decoderInfoPerfo(provider, DEVICE, LOGICAL, XBASIS, ROUNDS, file_name, threshold=0.01):
    md = load_metadata(DEVICE, LOGICAL, XBASIS, ROUNDS)
    jobs_by_calib_date = organize_jobs_by_calibration_date(md)

    idx = 0
    for calib_date, job_ids in jobs_by_calib_date.items():
        idx += 1
        if idx > 2:
            break
        
        kde_dict, kde_dict_PS, msmt_error_dict, noise_dict = retrieve_calib_data(provider, DEVICE, calib_date)
        big_memory, d, T, link_qubits, code_qubits, shots_per_job = retrieve_memories_and_job_info(provider, jobs_by_calib_date[calib_date], md)
        inverted_q_map = inv_qubit_mapping(get_repcode_IQ_map(link_qubits+code_qubits, T))

        # pSoft, ratio, countMat, pSoft_PS, ratio_PS, countMat_PS = get_pSoft_and_countMat(big_memory, kde_dict, kde_dict_PS, inverted_q_map, threshold, plot = True)
        pSoft_PS, ratio_PS, countMat_PS = get_pSoft_and_countMat(big_memory, kde_dict, kde_dict_PS, inverted_q_map, threshold, plot = True)
        # pSoft_mean = np.mean(pSoft)
        pSoft_PS_mean = np.mean(pSoft_PS)

        # distances = np.arange(3, d+1, 6)[::-1] # 6 stepsize to have 3x lower 
        # distances = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 27, 33, 39, 45, d]
        distances = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 27, 33, 39]
        for d_small in tqdm(distances, desc=f"distance"):
            # pSoft_subset_big, countMat_subset_big, num_subsets, num_shots_subset_tot = get_big_subset_mats(d_small, T, d, pSoft, countMat)
            pSoft_subset_big_PS, countMat_subset_big_PS, num_subsets_PS, num_shots_PS_subset_tot = get_big_subset_mats(d_small, T, d, pSoft_PS, countMat_PS)

            # assert num_shots_subset_tot == num_shots_PS_subset_tot, "num_shots and num_shots_PS should be equal"
            
            _RESETS = False
            model_mean_for_soft, model_mean_for_hard, model_mean_mean_for_hard, model_mean_mean_for_hard_PS, noise_list = get_stim_models(noise_dict, msmt_error_dict, 0.4, pSoft_PS_mean, d_small,
                                                                                                                      d, T, LOGICAL, XBASIS, _RESETS, link_qubits, code_qubits)
            
            nBits_res_dict = {}
            nBits_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, -1]
            # nBits_range = list(range(1, 12, 2))
            for nBits in nBits_range:
                if nBits == -1:
                    # pSoft_subset_big_quantized = pSoft_subset_big
                    pSoft_subset_big_PS_quantized = pSoft_subset_big_PS
                else:
                    # pSoft_subset_big_quantized = csi.quantizeMatrixVectorized(pSoft_subset_big, nBits)
                    pSoft_subset_big_PS_quantized = csi.quantizeMatrixVectorized(pSoft_subset_big_PS, nBits)

                # res_s_K = csi.decodeConvertorAll(model_mean_for_soft, countMat_subset_big, pSoft_subset_big_quantized, T, 
                                                    # int(LOGICAL), _RESETS, decode_hard=False)

                res_s_KPS = csi.decodeConvertorAll(model_mean_for_soft, countMat_subset_big_PS, pSoft_subset_big_PS_quantized, T,
                                                                int(LOGICAL), _RESETS, decode_hard=False)

                result_dict = {
                    # 'dict_s_K': res_to_job_subset_res(res_s_K, shots_per_job, num_subsets, job_ids),
                    'dict_s_KPS': res_to_job_subset_res(res_s_KPS, shots_per_job, num_subsets_PS, job_ids),
                }
                for method, job_ids_dict in result_dict.items():
                    for job_id, err_per_subset_list in job_ids_dict.items():
                        if method not in nBits_res_dict:
                            nBits_res_dict[method] = {}
                        if job_id not in nBits_res_dict[method]:
                            nBits_res_dict[method][job_id] = []
                        nBits_res_dict[method][job_id].append(sum(err_per_subset_list))
                        

            if not os.path.exists(file_name):
                data = {}
            else:
                with open(file_name, "r") as f:
                    data = json.load(f)

            for job_id in job_ids:
                if job_id not in data:
                    data[job_id] = {"additional_info": {
                        'threshold': threshold,
                        # 'ratio_leaked': ratio,
                        'ratio_leaked_PS': ratio_PS,
                        'noise_list': [float(noise) for noise in noise_list],
                        'nBits_list': nBits_range,
                    }, "distances": {}}

                if str(d_small) not in data[job_id]['distances']:
                    data[job_id]["distances"][str(d_small)] = {'tot_shots_with_all_subsets': float(num_shots_PS_subset_tot/len(job_ids))}

                for method, res_dict in nBits_res_dict.items():
                    data[job_id]['distances'][str(d_small)][method] = {
                        'errs_per_bit': res_dict[job_id]
                    }

            with open(file_name, "w") as f:
                json.dump(data, f, indent=4)






def load_metadata(DEVICE, LOGICAL, XBASIS, ROUNDS):
    while True:
        try:
            md = metadata_loader(True, True)
            break
        except:
            sleep(5)
    md = md.dropna(subset=["rounds", "distance", "code", "meas_level"])
    md['rounds'] = md['rounds'].astype(int)
    md['distance'] = md['distance'].astype(int)
    md = md[(md["job_status"] == "JobStatus.DONE") & 
            (md["code"] == "RepetitionCodeCircuit") & 
            (md["descr"] == 'subset RepCodes') &
            (md["meas_level"] == 1) &
            (md["backend_name"] == DEVICE) &
            (md["logical"] == LOGICAL) &
            (md["xbasis"] == XBASIS) &
            (md["rounds"] == ROUNDS) ]
    
    md = md[pd.isna(md["resets"])]
    
    # md = md[:1]
    print(f"shape md before: {md.shape}")
    md = md[:20] # take the first 20 jobs for first weekend!


    state = 'X' if XBASIS else 'Z'
    print(f"State: {state}{LOGICAL} {ROUNDS}")
    print(f"shape md: {md.shape}")
  
    return md


def organize_jobs_by_calibration_date(md):
    jobs_by_calibration_date = {}
    for _, row in md.iterrows():
        job_id = row['job_id']
        while True:
            try:
                _, _, calib_creation_date = find_closest_calib_jobs(tobecalib_job=job_id, verbose=False, double_msmt=False)
                break
            except:
                sleep(5)
        jobs_by_calibration_date.setdefault(calib_creation_date, []).append(job_id)
    print(f"Calibration dates: {jobs_by_calibration_date.keys()}")
    print(f"Num of calibrations: {len(jobs_by_calibration_date)}")
    print(f"Num of jobs per calibration: {[len(jobs) for jobs in jobs_by_calibration_date.values()]}\n")
    return jobs_by_calibration_date


def retrieve_calib_data(provider, DEVICE, calib_date, bandwidths=[0.6], rel_error=1, num_points=20, return_gmm=False):
    if rel_error != 1:
        print(f"\nWARNING: rel_error: {rel_error}\n")
    noise_dict = get_noise_dict_from_backend(provider, DEVICE, date = calib_date)
    while True:
        try:
            all_memories, gmm_dict, _ = load_calibration_memory(provider, tobecalib_backend=DEVICE, 
                                                                        other_date=calib_date, post_process=True,
                                                                        double_msmt=False)
            all_memories_PS, _, msmt_err_dict_PS = load_calibration_memory(provider, tobecalib_backend=DEVICE, 
                                                                                other_date=calib_date, post_process=True,
                                                                                double_msmt=True)
            break
        except:
            sleep(5)

    kde_dict = csi.get_KDEs(all_memories, bandwidths, relError=rel_error, absError=-1, num_points=num_points)
    kde_dict_PS = csi.get_KDEs(all_memories_PS, bandwidths, relError=rel_error, absError=-1, num_points=num_points)

    if return_gmm:
        return kde_dict, kde_dict_PS, msmt_err_dict_PS, noise_dict, gmm_dict
    else:
        return kde_dict, kde_dict_PS, msmt_err_dict_PS, noise_dict


def retrieve_memories_and_job_info(provider, job_ids, md):
    big_memory = np.vstack([provider.retrieve_job(job_id).result().get_memory() for job_id in job_ids])
    d = md[md["job_id"] == job_ids[0]]["distance"].values[0]
    T = md[md["job_id"] == job_ids[0]]["rounds"].values[0]
    shots_per_job = md[md["job_id"] == job_ids[0]]["shots"].values[0]

    job_sgl = provider.retrieve_job(job_ids[0])
    layout_des = job_sgl.deserialize_layout(job_sgl.initial_layouts()[0])
    link_qubits = list(layout_des['link_qubit'].values())
    code_qubits = list(layout_des['code_qubit'].values())

    return big_memory, d, T, link_qubits, code_qubits, shots_per_job


def get_pSoft_and_countMat(big_memory, kde_dict, kde_dict_PS, inverted_q_map, threshold=0.01, rel_error=1, plot = False):
    if rel_error != 1:
        print(f"\nWARNING: rel_error: {rel_error}\n")

    # print(f"\nStarting to get pSoft at {datetime.now()}")
    # pSoft, countMat, estim0Mat, estim1Mat = csi.iqConvertorEstim(big_memory, inverted_q_map, kde_dict, rel_error, -1)
    # print(f"Starting to get pSoft_PS at {datetime.now()}")
    pSoft_PS, countMat_PS, estim0MatPS, estim1MatPS = csi.iqConvertorEstim(big_memory, inverted_q_map, kde_dict_PS, rel_error, -1)
    print(f"Finished getting pSofts at {datetime.now()}\n")

    # pSoft, ratio = process_pSoft(pSoft, estim0Mat, estim1Mat, threshold)
    pSoft_PS, ratio_PS = process_pSoft(pSoft_PS, estim0MatPS, estim1MatPS, threshold)

    if plot:
        for qubit in [3, 69]:
            cols = inverted_q_map[qubit]
            # mask_1 = (pSoft[:, cols] == 0.5-1e-8)
            maskPS_1 = (pSoft_PS[:, cols] == 0.5-1e-8)
            # print(f"sum mask_1: {np.sum(mask_1)}")
            print(f"sum maskPS_1: {np.sum(maskPS_1)}")
            # plot_IQ_data_with_countMat(big_memory[:, cols][mask_1], countMat[:, cols][mask_1], title=f"not PS | qubit {qubit}, threshold {threshold:.0%}",
            #                            figsize=(5, 3))
            plot_IQ_data_with_countMat(big_memory[:, cols][maskPS_1], countMat_PS[:, cols][maskPS_1], title=f"PS | qubit {qubit}, threshold {threshold:.0%}",
                                       figsize=(5, 3))

    # return pSoft, ratio, countMat, pSoft_PS, ratio_PS, countMat_PS
    return  pSoft_PS, ratio_PS, countMat_PS


def get_big_subset_mats(d_small, T, d, pSoft, countMat):
    subsets = generate_subsets_pyramid(d, d_small)
    # subsets = generate_subsets_with_center(d, d_small)
    pSoft_subset_big = np.vstack([pSoft[:, get_cols_to_keep(subset, T, d)] for subset in subsets]) 
    countMat_subset_big = np.vstack([countMat[:, get_cols_to_keep(subset, T, d)] for subset in subsets])

    num_subsets = len(subsets)
    num_shots = countMat_subset_big.shape[0]

    return pSoft_subset_big, countMat_subset_big, num_subsets, num_shots


def get_stim_models(noise_dict, msmt_err_dict, pSoft_mean, pSoft_PS_mean, d_small, d, T, LOGICAL, XBASIS, _RESETS, link_qubits, code_qubits, noise_list=None):
    if noise_list is None:
        layout = link_qubits + code_qubits
        avgs = get_avgs_from_dict(noise_dict, layout)
        used_msmt_err_dict = {key: value for key, value in msmt_err_dict.items() if key in layout}
        ps_mean = sum(value['p_soft'] for value in used_msmt_err_dict.values()) / len(used_msmt_err_dict)
        ph_mean = sum(value['p_hard'] for value in used_msmt_err_dict.values()) / len(used_msmt_err_dict)
        noise_list = [avgs['two_gate'], avgs['single_gate'], avgs['t1_err'], avgs['t2_err'], float((ps_mean + ph_mean)), ph_mean, ps_mean]

    subsampling = d_small < d
    code_mean_for_soft = RepetitionCodeStimCircuit(d_small, T, XBASIS, _RESETS, noise_list=noise_list,
                                                            subsampling=subsampling, no_fin_soft=True, layout=None,
                                                            msmt_err_dict=None)
    model_mean_for_soft = code_mean_for_soft.circuits[LOGICAL].detector_error_model()
    code_mean_for_hard = RepetitionCodeStimCircuit(d_small, T, XBASIS, _RESETS, noise_list=noise_list,
                                                            subsampling=subsampling, no_fin_soft=False, layout=None,
                                                            msmt_err_dict=None)
    model_mean_for_hard = code_mean_for_hard.circuits[LOGICAL].detector_error_model()
    
    new_noise_list = noise_list.copy()
    new_noise_list[-1] = pSoft_mean
    code_mean_mean_for_hard = RepetitionCodeStimCircuit(d_small, T, XBASIS, _RESETS, noise_list=new_noise_list,
                                                            subsampling=subsampling, no_fin_soft=False, layout=None,
                                                            msmt_err_dict=None)
    model_mean_mean_for_hard = code_mean_mean_for_hard.circuits[LOGICAL].detector_error_model()

    new_noise_list[-1] = pSoft_PS_mean
    code_mean_mean_for_hard_PS = RepetitionCodeStimCircuit(d_small, T, XBASIS, _RESETS, noise_list=new_noise_list,
                                                            subsampling=subsampling, no_fin_soft=False, layout=None,
                                                            msmt_err_dict=None)
    model_mean_mean_for_hard_PS = code_mean_mean_for_hard_PS.circuits[LOGICAL].detector_error_model()
    


    return model_mean_for_soft, model_mean_for_hard, model_mean_mean_for_hard, model_mean_mean_for_hard_PS, noise_list


def res_to_job_subset_res(result, shots_per_job, num_subsets, job_ids):
    num_errs_per_job = {job_id: [0]*num_subsets for job_id in job_ids}
    num_jobs = len(job_ids)
    for err_idx in result.indices:
        subset_idx = err_idx // (shots_per_job * num_jobs)  # Corrected formula for subset index calculation
        job_idx = (err_idx % (shots_per_job * num_jobs)) // shots_per_job
        job_id = job_ids[int(job_idx)]
        num_errs_per_job[job_id][int(subset_idx)] += 1

    assert np.sum([np.sum(errs) for errs in num_errs_per_job.values()]) == len(result.indices), "Sum of errors should be equal to the number of errors"
    assert len(result.indices) == result.num_errors

    return num_errs_per_job


def process_pSoft(pSoft, estim0Mat, estim1Mat, threshold):
    pSoft_copy = pSoft.copy()
    mask = ((estim0Mat < threshold) & (estim1Mat < threshold))
    pSoft_copy[mask] = 0.5-1e-8

    # calculate the filtered ratio
    ratio = np.sum(mask) / (pSoft.shape[0]*pSoft.shape[1])
    return pSoft_copy, ratio


def generate_subsets_pyramid(d, d_small):
    og_sbs = list(range(d_small))
    subsets = []
    for i in range(0, d-d_small+1):
        subsets.append([x+i for x in og_sbs])

    return subsets