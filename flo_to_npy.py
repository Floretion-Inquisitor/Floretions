import numpy as np
import os
from multiprocessing import Pool, cpu_count
from floretion import Floretion



"""
This module computes and saves indices and signs matrices for floretion multiplication,
facilitating efficient computation of high-order floretion products.
"""


def compute_table_segment(args):
    """
    Computes indices and signs matrices for a segment of base vectors and stores as numpy binary files npy

    Args:
        args (tuple): Tuple containing base vectors, index range, floretion order, and segment number.

    Returns:
        str: Success message indicating that the segment has been processed and saved.
    """
    base_vecs, index_range, flo_order, segment = args
    unit_flo = Floretion.from_string(f'1{"e" * flo_order}')
    all_flo = Floretion(np.ones(4 ** flo_order), unit_flo.base_vec_dec_all, format_type="dec")
    num_base_vecs = 4 ** flo_order

    results_ind = np.zeros((len(index_range), num_base_vecs), dtype="int32")
    results_sgn = np.zeros((len(index_range), num_base_vecs), dtype="int32")

    for i, index_main in enumerate(index_range):
        z = base_vecs[index_main]
        result_ind_array = []
        result_sgn_array = []

        for index_y, y in enumerate(unit_flo.base_vec_dec_all):
            x = Floretion.mult_flo_base_absolute_value(z, y, flo_order)
            index_x = all_flo.base_to_grid_index[x]
            sign_xy = Floretion.mult_flo_sign_only(x, y, flo_order)

            result_ind_array.append(index_x)
            result_sgn_array.append(sign_xy)

        results_ind[i] = result_ind_array
        results_sgn[i] = result_sgn_array

    # Save directly to disk
    file_name_ind = f'{filedir}/floretion_order_{flo_order}_segment_{segment}_indices.npy'
    file_name_sgn = f'{filedir}/floretion_order_{flo_order}_segment_{segment}_signs.npy'
    np.save(file_name_ind, results_ind)
    np.save(file_name_sgn, results_sgn)

    return f"Segment {segment} processed and saved."


if __name__ == "__main__":

    # suggestions
    # flo_order = 8 : total_segments = 32
    # flo_order = 7 : total_segments = 4
    # flo_order = < 7 : total_segments = 1

    flo_order = 1
    filedir = f"./data/npy/order_{flo_order}"
    os.makedirs(filedir, exist_ok=True)

    total_segments = 64

    # change depending
    cores_per_batch = 8

    num_base_vecs = 4 ** flo_order
    base_vecs = Floretion.from_string(f'1{"e" * flo_order}').base_vec_dec_all


    do_batches = False
    if do_batches:
        for batch_start in range(0, total_segments, cores_per_batch):
            with Pool(processes=cores_per_batch) as pool:
                tasks = [(base_vecs, range(segment * (num_base_vecs // total_segments),
                                           (segment + 1) * (num_base_vecs // total_segments)),
                          flo_order, segment)
                         for segment in range(batch_start, min(batch_start + cores_per_batch, total_segments))]

                results = pool.map(compute_table_segment, tasks)

            for result in results:
                print(result)

        print("All computations and storage completed.")

