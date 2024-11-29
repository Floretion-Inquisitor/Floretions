# MIT License

# Copyright (c) [2024 [Creighton Dement]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pandas as pd
import re
import os
import json
from multiprocessing import Pool, cpu_count
import floretion_base_vector
import logging
import time

logger = logging.getLogger('FloretionLogger')

# Set the level of the logger. This can be DEBUG, INFO, WARNING, ERROR, or CRITICAL.
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('floretion.log')

# Set levels for handlers
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLORETION_VERSION = 1.01
DEFAULT_PROCESS_COUNT = max(2, (3 * cpu_count() // 4) // 2 * 2)


def count_bits(n):
    """Count the number of '1's in the binary representation of an integer."""
    return bin(n).count('1')


def sgn(x):
    """Return the sign of a number: -1 for negative, 1 for non-negative."""
    return -1 if x < 0 else 1


class Floretion:
    """
    Represents a Floretion (see ) with base vectors defined in octal format
    using digits from {1, 2, 4, 7}. These base vectors represent the group of all floretions of order n,
    where each floretion uses XOR for multiplication. Data about the floretion group is imported from
    'floretions/data', f"grid.flo_{self.flo_order}.oct.csv" upon initialization.

    Examples:
        - First Order Floretion (Quaternion):
          flo_x = Floretion.from_string('1i+2j+3k+4e')
          flo_y = Floretion.from_string('k-j+3')
          flo_z = 2 * flo_x * flo_y

        - Third Order Floretion:
          flo_x = Floretion.from_string('1iii-3iji +2jkj - 4iee')
          flo_y = Floretion.from_string('6iijk- 4ieek + kkke')
          flo_z = flo_x * flo_y + 2 * flo_y

        - Direct Initialization with Arrays:
          flo_x = Floretion([1, 0.5, -0.2, 3], [11, 21, 41, 77], format_type="oct")
          # use as_floretion_notation() on any floretion for printing in floretion notation
          flo_x.as_floretion_notation() -> "ii + .5ji - 0.2ki + 3ee"

          commutes_ie = np.array([9, 10, 12, 15, 57, 58, 60, 63])
          flo_commutes_ie = Floretion(np.ones(commutes_ie.size), commutes_ie, format_type="dec")
          print(f"flo_commutes_ie {flo_commutes_ie.as_floretion_notation()}")

    Attributes:
        base_to_nonzero_coeff (dict): Mapping of base vectors (in decimal) to their non-zero coefficients.
        format_type (str): Format of base vectors ('dec' or 'oct').
        max_order (int): Maximum order for the floretions.
        flo_order (int): Current order of the floretion.
        grid_flo_loaded_data (DataFrame): Pre-loaded data.
        base_vec_dec_all (np.array): Array of all base vectors in decimal.
        coeff_vec_all (np.array): Array of coefficients aligned with `base_vec_dec_all`.
        base_to_grid_index (dict): Mapping of base vectors to their grid indices.
    """
    num_processes = DEFAULT_PROCESS_COUNT

    def __init__(self, coeffs_of_base_vecs, base_vecs, grid_flo_loaded_data=None, format_type="dec"):
        """
        Initializes a Floretion instance with given coefficients and base vectors, with optional pre-loaded data.
        Parameters:
            coeffs_of_base_vecs (list): Coefficients of the base vectors.
            base_vecs (list): Base vectors corresponding to the coefficients.
            grid_flo_loaded_data (DataFrame): Optional pre-loaded data, defaults to None.
            format_type (str): Indicates the format of base vectors ('dec' or 'oct'), defaults to 'dec'.
        """
        self.format_type = format_type
        temp_base_vec_dec = np.array([int(str(bv), 8) if format_type == "oct" else bv for bv in base_vecs])
        self.base_to_nonzero_coeff = {bv: coeff for bv, coeff in zip(temp_base_vec_dec, coeffs_of_base_vecs) if
                                      np.abs(coeff) > np.finfo(float).eps}
        # define maximum order of floretions for which base vectors are defined in input data
        self.max_order = 10
        self.flo_order = self.find_flo_order(temp_base_vec_dec, self.max_order)
        self.load_floretion_data(grid_flo_loaded_data)

    def load_floretion_data(self, grid_flo_loaded_data):
        """Load floretion data from file or use provided DataFrame."""
        file_path = os.path.join(BASE_DIR, 'floretions/data', f"grid.flo_{self.flo_order}.oct.csv")
        if grid_flo_loaded_data is None:
            try:
                self.grid_flo_loaded_data = pd.read_csv(file_path, dtype={'oct': str})
            except FileNotFoundError:
                logging.error(f"The file {file_path} could not be found.")
                raise
        else:
            self.grid_flo_loaded_data = grid_flo_loaded_data

        self.base_vec_dec_all = self.grid_flo_loaded_data['floretion'].to_numpy()
        self.coeff_vec_all = np.zeros_like(self.base_vec_dec_all, dtype=float)
        self.map_coeffs_to_base_vectors()

    def map_coeffs_to_base_vectors(self):
        """Map coefficients to base vectors using the pre-loaded grid data."""
        self.base_to_grid_index = {base_vec: idx for idx, base_vec in enumerate(self.base_vec_dec_all) if
                                   base_vec in self.base_to_nonzero_coeff}
        for base_vec, coeff in self.base_to_nonzero_coeff.items():
            idx = self.base_to_grid_index.get(base_vec)
            if idx is not None:
                self.coeff_vec_all[idx] = coeff
            else:
                logging.warning(f"Base vector {base_vec} not found in grid.")

    @classmethod
    def from_preloaded_data(cls, coeffs_of_base_vecs, base_vecs, flo_order, grid_flo_data):
        """
        Creates a Floretion instance from pre-loaded data.

        Parameters:
            coeffs_of_base_vecs: A list of coefficients.
            base_vecs: A list of base vectors.
            flo_order: The floretion order.
            grid_flo_data: A Pandas DataFrame of grid data.

        Returns:
            A new Floretion instance.
        """
        if len(base_vecs) != len(grid_flo_data):
            raise ValueError(
                f"The length of base_vecs {len(base_vecs)} must match the number of rows in grid_flo_data {len(grid_flo_data)}.")

        instance = cls.__new__(cls)

        # Assuming base_vecs and coeffs are non-zero and properly aligned
        instance.base_to_nonzero_coeff = dict(zip(base_vecs, coeffs_of_base_vecs))
        instance.flo_order = flo_order
        instance.grid_flo_loaded_data = grid_flo_data
        instance.base_vec_dec_all = np.array(base_vecs)
        instance.coeff_vec_all = np.array(coeffs_of_base_vecs)

        # base_to_grid_index can be directly computed if base_vec_dec_all and base_to_nonzero_coeff are the same
        instance.base_to_grid_index = {}
        for base_vec in base_vecs:
            index_row = grid_flo_data[grid_flo_data['floretion'] == base_vec].index[0]
            instance.base_to_grid_index[base_vec] = index_row

        return instance

    def find_flo_order(self, temp_base_vec_dec, max_order):
        """
        Determines the order of the floretion.

        Parameters
        ----------
        temp_base_vec_dec : np.array
            An array of base vectors in decimal representation.
        max_order : int
            The maximum order to be considered.

        Returns
        -------
        int
            The order of the floretion.
        """
        common_order = -1
        for base_element in temp_base_vec_dec:
            flo_order = 0
            found_order = False
            while flo_order <= max_order and not found_order:
                flo_order += 1
                if base_element <= (8 ** flo_order) - 1:
                    found_order = True

            if common_order == -1:
                common_order = flo_order
            elif common_order != flo_order:
                raise ValueError("All base vectors must have the same order")

        return common_order

    def __pow__(self, exponent):
        if exponent == 0:
            return Floretion(1)  # assuming a constructor that can create an identity element
        elif exponent == 1:
            return self
        elif exponent > 1:
            result = self
            for _ in range(exponent - 1):
                result *= self  # assuming you've already implemented __mul__
            return result
        else:
            raise ValueError("Exponent must be a non-negative integer for this example.")

    def __add__(self, other):
        new_coeffs = self.coeff_vec_all + other.coeff_vec_all
        return Floretion(new_coeffs, self.base_vec_dec_all, self.grid_flo_loaded_data)

    def __sub__(self, other):
        new_coeffs = self.coeff_vec_all - other.coeff_vec_all
        return Floretion(new_coeffs, self.base_vec_dec_all, self.grid_flo_loaded_data)

    def __eq__(self, other):
        """
        Check if this floretion is equal to another floretion.

        Parameters:
            other (Floretion): The floretion to compare with.

        Returns:
            bool: True if the floretions are equal, False otherwise.
        """
        if not isinstance(other, Floretion):
            # The other object is not a Floretion, so they can't be equal
            return False

        return np.array_equal(self.coeff_vec_all, other.coeff_vec_all)

    # First declare mult_flo_base_absolute_value
    @staticmethod
    def mult_flo_base_absolute_value(a_base_val, b_base_val, flo_order):
        """
        Computes the absolute value of the product of two floretion base vectors.

        Parameters:
            a_base_val: The first base vector.
            b_base_val: The second base vector.
            flo_order: The floretion order.

        Returns:
            An integer representing the absolute value of the floretion product.
        """
        bitmask = int(2 ** (3 * flo_order) - 1)
        a_base_val = abs(a_base_val)
        b_base_val = abs(b_base_val)

        return bitmask & (~(a_base_val ^ b_base_val))

    @staticmethod
    def mult_flo_sign_only(a_base_val, b_base_val, flo_order):
        """
        Computes only the sign of the product of two floretion base vectors.

        Parameters:
            a_base_val: The first base vector.
            b_base_val: The second base vector.
            flo_order: The floretion order.

        Returns:
            An integer representing the sign of the floretion product (-1 or 1).
        """
        bitmask = int(2 ** (3 * flo_order) - 1)
        oct666 = int('6' * flo_order, 8)
        oct111 = int('1' * flo_order, 8)

        pre_sign = sgn(a_base_val) * sgn(b_base_val)
        a_base_val = abs(a_base_val)
        b_base_val = abs(b_base_val)

        # Shift every 3-bits of "a" one to the left
        a_cyc = ((a_base_val << 1) & oct666) | ((a_base_val >> 2) & oct111)

        cyc_sign = 1 if count_bits((a_cyc & b_base_val) & bitmask) & 0b1 else -1
        ord_sign = 1 if count_bits(bitmask) & 0b1 else -1

        return (pre_sign * cyc_sign * ord_sign)

    @staticmethod
    def mult_flo_base_only(a_base_val, b_base_val, flo_order):
        """
        Computes only the sign of the product of two floretion base vectors.

        Parameters:
            a_base_val: The first base vector.
            b_base_val: The second base vector.
            flo_order: The floretion order.

        Returns:
            An integer representing the sign of the floretion product (-1 or 1).
        """
        bitmask = int(2 ** (3 * flo_order) - 1)
        oct666 = int('6' * flo_order, 8)
        oct111 = int('1' * flo_order, 8)

        pre_sign = sgn(a_base_val) * sgn(b_base_val)
        a_base_val = abs(a_base_val)
        b_base_val = abs(b_base_val)

        # Shift every 3-bits of "a" one to the left
        a_cyc = ((a_base_val << 1) & oct666) | ((a_base_val >> 2) & oct111)

        cyc_sign = 1 if count_bits((a_cyc & b_base_val) & bitmask) & 0b1 else -1
        ord_sign = 1 if count_bits(bitmask) & 0b1 else -1

        abs_val = bitmask & (~(a_base_val ^ b_base_val))
        sign = (pre_sign * cyc_sign * ord_sign)

        return abs_val * sign

    # Then declare compute_possible_vecs and other functions
    @staticmethod
    def compute_possible_vecs(x_base_vecs, y_base_vecs, flo_order, base_vecs_all):
        """
         Computes a set of possible vectors for multiplication.

         Parameters:
             x_base_vecs: A list of base vectors for the first floretion.
             y_base_vecs: A list of base vectors for the second floretion.
             flo_order: The floretion order.
             base_vecs_all: A list of all available base vectors.

         Returns:
             A set of possible base vectors.
         """
        possible_vecs = set()

        if len(x_base_vecs) > len(base_vecs_all) // 2:
            return set(base_vecs_all)

        for z in base_vecs_all:
            for x in x_base_vecs:
                product_abs = Floretion.mult_flo_base_absolute_value(x, z, flo_order)
                if product_abs in y_base_vecs:
                    possible_vecs.add(z)

        return possible_vecs

    @staticmethod
    def write_centers_to_file(flo_order, decomposition_type="pos"):

        dummy_flo = Floretion.from_string(f'0{"e" * flo_order}')
        base_vec_all = (dummy_flo.base_vec_dec_all)
        result = {}

        for base_vec in base_vec_all:
            this_flo = Floretion(np.array([1]), np.array([base_vec]))
            center = np.array(sorted(list(this_flo.find_center_base_vectors_only(decomposition_type))))
            result[base_vec] = center

        result_str_keys = {str(key): value.tolist() for key, value in result.items()}

        # Manually construct the JSON string
        json_lines = ["{"]
        last_key = list(result_str_keys.keys())[-1]
        for key, value in result_str_keys.items():
            # Convert each key-value pair to a JSON string
            json_pair = json.dumps({key: value})
            json_pair = json_pair.strip("{}")

            # Add a comma after each pair except the last one
            if key != last_key:
                json_pair += ","

            json_lines.append("    " + json_pair)
        json_lines.append("}")

        json_string = "\n".join(json_lines)

        # Write the formatted JSON string to a file
        with open(f'center_data_order_{flo_order}.{decomposition_type}.json', 'w') as file:
            file.write(json_string + "\n")  # Ensure there's a newline at the end of the file

        # Save to JSON file
        # with open(f'center_data_order_{flo_order}.json', 'w') as file:
        #    json.dump(json_string, file, separators="\n")
        # return result

    @staticmethod
    def load_centers(flo_order, decomposition_type="pos", storage_type="npy"):
        # i:\dev\dev\python\floretions\data\centers\order_6\both\centers_order_6_segment_000.npy
        centers_file_path = os.path.join(BASE_DIR, f'floretions/data/centers/order_{flo_order}/{decomposition_type}/',
                                         f'centers_order_{flo_order}_segment_000.{storage_type}')

        if storage_type == "npy":
            center_data = np.load(centers_file_path, allow_pickle=True)
        elif storage_type == "json":
            # Load center data
            with open(centers_file_path, 'r') as file:
                center_data = json.load(file)

        return center_data

    @staticmethod
    def multiply_segment(indices, signs, other_coeffs):
        # Compute products of coefficients for each segment using precomputed indices and signs
        x_coeffs_reordered = other_coeffs[indices] * signs

        return np.dot(x_coeffs_reordered, other_coeffs)

    def process_chunk(args):
        z_indices, x_coeff_vec_all, y_coeff_vec_all, indices_matrix, signs_matrix = args
        local_coeffs = []
        for z_index in z_indices:
            x_coeffs_reordered = x_coeff_vec_all[indices_matrix[z_index]] * signs_matrix[z_index]
            local_coeffs.append(np.dot(x_coeffs_reordered, y_coeff_vec_all))
        return local_coeffs

    def __mul__(self, other):
        """
        Overloads the multiplication (*) operator for Floretion instances.

        Parameters:
            other: A Floretion instance or a scalar (int, float) for scalar multiplication.

        Returns:
            A new Floretion instance resulting from the multiplication.

        """
        # Handle scalar multiplication
        if isinstance(other, (int, float)):
            new_coeff_vec_all = self.coeff_vec_all * other
            return Floretion(new_coeff_vec_all, self.base_vec_dec_all, self.grid_flo_loaded_data)

        # Handle floretion multiplication
        else:
            # An optimization step can be added here for low orders
            if False:  # self.flo_order < 4:
                possible_base_vecs = self.base_to_nonzero_coeff
            else:
                # Compute possible base vectors for optimization
                possible_base_vecs = self.compute_possible_vecs(self.base_to_nonzero_coeff,
                                                                other.base_to_nonzero_coeff,
                                                                self.flo_order,
                                                                self.base_vec_dec_all)

            z_base_vecs = list()
            z_coeffs = list()

            if self.flo_order > 5:
                do_table_import = True
            else:
                do_table_import = False

            if not do_table_import:
                # For each possible base vector 'z'
                for z in possible_base_vecs:
                    coeff_z = 0.0

                    # For each base vector 'y' of the other Floretion
                    for base_vec_y, coeff_y in other.base_to_nonzero_coeff.items():
                        # Compute absolute value product of base vectors
                        check_if_in_base_vec_x = Floretion.mult_flo_base_absolute_value(z, base_vec_y, self.flo_order)

                        if check_if_in_base_vec_x in self.base_to_nonzero_coeff.keys():
                            # Lookup coefficient for base vector x
                            index_x = self.base_to_grid_index[check_if_in_base_vec_x]
                            coeff_x = self.coeff_vec_all[index_x]

                            # Compute coefficient for the result base vector z
                            coeff_z += coeff_x * coeff_y * Floretion.mult_flo_sign_only(check_if_in_base_vec_x,
                                                                                        base_vec_y,
                                                                                        self.flo_order)

                    z_coeffs.append(coeff_z)
                    z_base_vecs.append(z)

                return Floretion(z_coeffs, z_base_vecs, self.grid_flo_loaded_data)

            else:

                filedir = f"./data/npy/order_{self.flo_order}"
                segment = "000"  # segment_string = str(segment).zfill(3)
                file_name_ind = f'{filedir}/floretion_order_{self.flo_order}_segment_{segment}_indices.npy'
                file_name_sgn = f'{filedir}/floretion_order_{self.flo_order}_segment_{segment}_signs.npy'

                indices_matrix = np.load(file_name_ind)
                signs_matrix = np.load(file_name_sgn)

                # print(indices_matrix[0])
                # print(signs_matrix)
                # For each possible base vector 'z'
                for z_index, z_base_vec in enumerate(self.base_vec_dec_all):
                    # print(f'z_index {z_index} z_base_vec {z_base_vec}')
                    # print(self.coeff_vec_all[indices_matrix[z_index]])
                    x_coeffs_reordered = self.coeff_vec_all[indices_matrix[z_index]] * signs_matrix[z_index]
                    z_coeffs.append(np.dot(x_coeffs_reordered, other.coeff_vec_all))

                return Floretion(z_coeffs, self.base_vec_dec_all, self.grid_flo_loaded_data)

    def mul_sp(self, other):
        if isinstance(other, (int, float)):
            new_coeff_vec_all = self.coeff_vec_all * other
            return Floretion(new_coeff_vec_all, self.base_vec_dec_all)
        else:
            if self.flo_order < 4:
                possible_base_vecs = self.base_vec_dec_all
            else:
                possible_base_vecs = self.compute_possible_vecs(self.base_to_nonzero_coeff, other.base_to_nonzero_coeff,
                                                                self.flo_order, self.base_vec_dec_all)

            # print(f" possible_base_vecs: {possible_base_vecs}")

            z_base_vecs = list()
            z_coeffs = list()
            # For each possible base vector 'z'
            for z in possible_base_vecs:
                coeff_z = 0.0

                for base_vec_y, coeff_y in other.base_to_nonzero_coeff.items():
                    check_if_in_base_vec_x = Floretion.mult_flo_base_absolute_value(z, base_vec_y, self.flo_order)
                    # print(f" check_if_in_base_vec_x {check_if_in_base_vec_x}")

                    if check_if_in_base_vec_x in self.base_to_nonzero_coeff.keys():
                        # index_y = other.base_to_grid_index[base_vec_y]
                        # coeff_y = other.coeff_vec_all[index_y]

                        index_x = self.base_to_grid_index[check_if_in_base_vec_x]
                        coeff_x = self.coeff_vec_all[index_x]
                        coeff_z += coeff_x * coeff_y * floretion_base_vector.mult_flo_sign_only(check_if_in_base_vec_x,
                                                                                                base_vec_y,
                                                                                                self.flo_order)
                        # print(f" coeff_z {coeff_z}  coeff_c {coeff_x} coeff_y {coeff_y} sign = {floretion_base_vector.mult_flo_sign_only(check_if_in_base_vec_x, base_vec_y, self.flo_order)}!")

                # print(f" z_coeff {z_coeff}!")
                z_coeffs.append(coeff_z)
                z_base_vecs.append(z)

            return z_coeffs

    def mul_sp(self, other):
        if isinstance(other, (int, float)):
            new_coeff_vec_all = self.coeff_vec_all * other
            return Floretion(new_coeff_vec_all, self.base_vec_dec_all)
        else:
            if self.flo_order < 4:
                possible_base_vecs = self.base_vec_dec_all
            else:
                possible_base_vecs = self.compute_possible_vecs(self.base_to_nonzero_coeff, other.base_to_nonzero_coeff,
                                                                self.flo_order, self.base_vec_dec_all)

            # print(f" possible_base_vecs: {possible_base_vecs}")

            z_base_vecs = list()
            z_coeffs = list()
            # For each possible base vector 'z'
            for z in possible_base_vecs:
                coeff_z = 0.0

                for base_vec_y, coeff_y in other.base_to_nonzero_coeff.items():
                    check_if_in_base_vec_x = Floretion.mult_flo_base_absolute_value(z, base_vec_y, self.flo_order)
                    # print(f" check_if_in_base_vec_x {check_if_in_base_vec_x}")

                    if check_if_in_base_vec_x in self.base_to_nonzero_coeff.keys():
                        # index_y = other.base_to_grid_index[base_vec_y]
                        # coeff_y = other.coeff_vec_all[index_y]

                        index_x = self.base_to_grid_index[check_if_in_base_vec_x]
                        coeff_x = self.coeff_vec_all[index_x]
                        coeff_z += coeff_x * coeff_y * floretion_base_vector.mult_flo_sign_only(check_if_in_base_vec_x,
                                                                                                base_vec_y,
                                                                                                self.flo_order)
                        # print(f" coeff_z {coeff_z}  coeff_c {coeff_x} coeff_y {coeff_y} sign = {floretion_base_vector.mult_flo_sign_only(check_if_in_base_vec_x, base_vec_y, self.flo_order)}!")

                # print(f" z_coeff {z_coeff}!")
                z_coeffs.append(coeff_z)
                z_base_vecs.append(z)

            return z_coeffs

    def __rmul__(self, scalar):
        if isinstance(scalar, (int, float)):
            new_coeff_vec_all = self.coeff_vec_all * scalar
            return Floretion(new_coeff_vec_all, self.base_vec_dec_all)

    def as_floretion_notation(self):
        floretion_terms = []

        for coeff, base_vec in zip(self.coeff_vec_all, self.base_vec_dec_all):
            # Create a string for the coefficient
            if coeff == 1:
                coeff_str = "+"
            elif coeff == -1:
                coeff_str = "-"
            else:
                coeff_str = f"{coeff:+.4f}"

            # Convert base-8 digits back to floretion symbols
            base_vec_str = ""
            base_vec_copy = base_vec
            while base_vec_copy > 0:
                digit = base_vec_copy & 7
                if digit == 1:
                    base_vec_str = 'i' + base_vec_str
                elif digit == 2:
                    base_vec_str = 'j' + base_vec_str
                elif digit == 4:
                    base_vec_str = 'k' + base_vec_str
                elif digit == 7:
                    base_vec_str = 'e' + base_vec_str
                base_vec_copy >>= 3

            # Assemble the term
            term = f"{coeff_str}{base_vec_str}"

            if coeff != 0:
                floretion_terms.append(term.strip())

        result_string = " ".join(floretion_terms).replace(" + -", " - ").replace(" + +", " + ")
        if result_string == "":
            result_string = " _0_"
        return result_string

    def tes(self):
        return self.coeff_vec_all[-1]

    @staticmethod
    def normalize_coeffs(floretion, max_abs_value=2.0):
        """
        Normalize the coefficients of a given Floretion instance and return a new instance.

        Args:
            floretion (Floretion): The Floretion instance to normalize.
            max_abs_value (float): The desired maximum absolute value of the coefficients.

        Returns:
            Floretion: New Floretion instance with normalized coefficients.
        """
        max_coeff = np.max(np.abs(floretion.coeff_vec_all))
        if max_coeff != 0:  # Avoid division by zero
            normalized_coeff_vec_all = max_abs_value * floretion.coeff_vec_all / max_coeff
        else:
            normalized_coeff_vec_all = floretion.coeff_vec_all

        # Create a new Floretion instance with normalized coefficients
        return Floretion(normalized_coeff_vec_all, floretion.base_vec_dec_all, floretion.grid_flo_loaded_data)

    @staticmethod
    def fractional_coeffs(floretion, max_abs_value=2.0):
        """

        """
        frac_coeffs = floretion.coeff_vec_all - np.round(floretion.coeff_vec_all, 0)

        # Create a new Floretion instance with normalized coefficients
        return Floretion(frac_coeffs, floretion.base_vec_dec_all, floretion.grid_flo_loaded_data)

    @staticmethod
    def mirror(floretion, axis):
        """
        Mirror the Floretion instance across the specified axis.

        Args:
            floretion (Floretion): The Floretion instance to mirror.
            axis (str): The axis to mirror across ('I', 'J', or 'K').

        Returns:
            Floretion: New Floretion instance mirrored across the specified axis.

        Raises:
            ValueError: If the axis is not 'I', 'J', or 'K'.
        """
        if axis not in ["I", "J", "K"]:
            raise ValueError("Axis must be 'I', 'J', or 'K'.")

        new_coeffs = np.zeros_like(floretion.coeff_vec_all)
        for coeff, base_vec in zip(floretion.coeff_vec_all, floretion.base_vec_dec_all):
            # Convert base_vec to octal and perform digit swaps based on the axis
            octal_str = format(base_vec, 'o')
            if axis == "I":
                octal_str = octal_str.replace('2', 'x').replace('4', '2').replace('x', '4')
            elif axis == "J":
                octal_str = octal_str.replace('1', 'x').replace('4', '1').replace('x', '4')
            elif axis == "K":
                octal_str = octal_str.replace('1', 'x').replace('2', '1').replace('x', '2')

            # Convert back to decimal
            new_base_vec = int(octal_str, 8)
            # Find the index of the new base vector
            idx = np.where(floretion.base_vec_dec_all == new_base_vec)[0]
            # Set the corresponding coefficient
            if idx.size > 0:
                new_coeffs[idx[0]] = coeff

        # Create a new Floretion instance with the new coefficients
        return Floretion(new_coeffs, floretion.base_vec_dec_all, floretion.grid_flo_loaded_data)

    def sum_of_squares(self):
        return sum(coeff ** 2 for coeff in self.coeff_vec_all)

    def abs(self):
        return np.sqrt(self.sum_of_squares())

    @staticmethod
    def decimal_to_octal(decimal):
        return format(int(decimal), 'o')

    @staticmethod
    def get_typical_floretions(name, order):
        print(f"Creating floretion {name} of order {order}.")

        # Initialize the base floretion and unit only once
        zero_flo = 0 * Floretion.from_string(f'1{"e" * order}')
        unit_flo = Floretion.from_string(f'1{"e" * order}')


        # Create base lists for coefficients
        new_coeffs_sierp = []
        new_coeffs_sierp_i = []
        new_coeffs_sierp_j = []
        new_coeffs_sierp_k = []

        axis_i = []
        axis_j = []
        axis_k = []

        # Calculate coefficients based on the zero floretion's base vectors
        for base in zero_flo.base_vec_dec_all:
            base_octal = Floretion.decimal_to_octal(base)

            new_coeffs_sierp.append(0.0 if '7' in base_octal else 1.0)
            new_coeffs_sierp_i.append(0.0 if '1' in base_octal else 1.0)
            new_coeffs_sierp_j.append(0.0 if '2' in base_octal else 1.0)
            new_coeffs_sierp_k.append(0.0 if '4' in base_octal else 1.0)

            axis_i.append(0.0 if '2' in base_octal or '4' in base_octal else 1.0)
            axis_j.append(0.0 if '4' in base_octal or '1' in base_octal else 1.0)
            axis_k.append(0.0 if '1' in base_octal or '2' in base_octal else 1.0)

        # Convert lists to NumPy arrays for fast array operations
        new_coeffs_sierp = new_coeffs_sierp
        new_coeffs_sierp_i = new_coeffs_sierp_i
        new_coeffs_sierp_j = new_coeffs_sierp_j
        new_coeffs_sierp_k = new_coeffs_sierp_k

        coeffs_axis_i = axis_i
        coeffs_axis_j = axis_j
        coeffs_axis_k = axis_k

        norm_fac = np.sqrt(1. / (order + 1) ** 3)

        # Define the dictionary for lazy-loaded floretion instances
        name_to_floretion = {
            "zero_flo": lambda: zero_flo,
            "unit_flo": lambda: unit_flo,
            "uniform_flo": lambda: Floretion(np.ones(zero_flo.base_vec_dec_all.size), zero_flo.base_vec_dec_all, format_type="dec"),
            "sierp_flo": lambda: Floretion(norm_fac * new_coeffs_sierp, zero_flo.base_vec_dec_all, format_type="dec"),
            "sierp_flo_i": lambda: Floretion(norm_fac * new_coeffs_sierp_i, zero_flo.base_vec_dec_all,
                                             format_type="dec"),
            "sierp_flo_j": lambda: Floretion(norm_fac * new_coeffs_sierp_j, zero_flo.base_vec_dec_all,
                                             format_type="dec"),
            "sierp_flo_k": lambda: Floretion(norm_fac * new_coeffs_sierp_k, zero_flo.base_vec_dec_all,
                                             format_type="dec"),
            "axis_ijk": lambda: (Floretion(coeffs_axis_i, zero_flo.base_vec_dec_all, format_type="dec") +
                                 Floretion(coeffs_axis_j, zero_flo.base_vec_dec_all, format_type="dec") +
                                 Floretion(coeffs_axis_k, zero_flo.base_vec_dec_all, format_type="dec") - 2 * unit_flo),
            "axis_ij": lambda: (Floretion(coeffs_axis_i, zero_flo.base_vec_dec_all, format_type="dec") +
                                Floretion(coeffs_axis_j, zero_flo.base_vec_dec_all, format_type="dec") - unit_flo),
            "axis_jk": lambda: (Floretion(coeffs_axis_j, zero_flo.base_vec_dec_all, format_type="dec") +
                                Floretion(coeffs_axis_k, zero_flo.base_vec_dec_all, format_type="dec") - unit_flo),
            "axis_ki": lambda: (Floretion(coeffs_axis_k, zero_flo.base_vec_dec_all, format_type="dec") +
                                Floretion(coeffs_axis_i, zero_flo.base_vec_dec_all, format_type="dec") - unit_flo)
        }

        # Lazy initialize the floretion based on the 'name'
        if name in name_to_floretion:
            result = name_to_floretion[name]()  # Call the lambda function to initialize the floretion
        else:
            result = None  # Or raise an error if the name is unknown

        return result

    @classmethod
    def from_string(cls, flo_string, format_type="dec"):

        flo_string = flo_string.replace(" ", "")

        if not all(c in "0123456789ijke.+-()" for c in flo_string):
            raise ValueError("Invalid character in floretion string. E.g. 3rd order: 3.5iii + jjk - 0.5eee (do not use '*')")

        flo_terms = flo_string.split(")(")

        # Error check 1: No invalid characters


        # Error check 2: No invalid signs
        if "++" in flo_string or "+-" in flo_string or "-+" in flo_string or "--" in flo_string:
            raise ValueError("Invalid sign combination in floretion string.")

        flo_string = flo_string.replace(" ", "")
        terms_str = re.findall(r'[\+\-]?[0-9]*\.?[0-9]*[ijke]+', flo_string)

        coeffs = []
        base_vecs = []

        for term in terms_str:
            match = re.match(r'([\+\-]?[0-9]*\.?[0-9]*)?([ijke]+)', term)
            if match:
                coeff_str, base_vec_str = match.groups()
                coeff = float(coeff_str) if coeff_str and coeff_str != '-' and coeff_str != '+' else 1.0
                if coeff_str and coeff_str[0] == '-':
                    coeff = -1.0 if coeff_str == '-' else coeff
                if coeff_str and coeff_str[0] == '+':
                    coeff = 1.0 if coeff_str == '+' else coeff

                base_vec = 0
                for ch in base_vec_str:
                    if ch == 'i':
                        base_vec = (base_vec << 3) | 1
                    elif ch == 'j':
                        base_vec = (base_vec << 3) | 2
                    elif ch == 'k':
                        base_vec = (base_vec << 3) | 4
                    elif ch == 'e':
                        base_vec = (base_vec << 3) | 7
                    else:
                        raise ValueError(f"Invalid character {ch} in floretion string.")

                coeffs.append(coeff)
                base_vecs.append(base_vec)
            else:
                raise ValueError(f"Invalid term '{term}' in floretion string.")

        return cls(coeffs_of_base_vecs=np.array(coeffs), base_vecs=np.array(base_vecs), format_type=format_type)

    def inverse(self):
        # Check if flo is a single base floretion (positive or negative)
        non_zero_elements = [coeff for coeff in self.coeff_vec_all if coeff != 0]
        if len(non_zero_elements) == 1 and abs(non_zero_elements[0]) == 1:
            # flo is a base floretion
            if self * self == Floretion.from_string("e"):
                return self  # Inverse is the floretion itself
            else:
                return -1 * self  # Inverse is the negative of the floretion
        else:
            raise ValueError("Floretion is not a base floretion.")

    def find_center(self):
        """
        Find all base vectors that commute with this floretion.

        Returns:
            A list of base vectors that commute with this floretion.
        """
        commuting_base_vectors = []

        for base_vec in self.base_vec_dec_all:
            # Initialize the base vector as a Floretion object
            base_floretion = Floretion([1], [base_vec], format_type="dec")

            # Check if the base vector commutes with this floretion
            if self * base_floretion == base_floretion * self:
                commuting_base_vectors.append(base_vec)

        return commuting_base_vectors

    def centered_flo(self, decomposition_type="both"):

        # initialize to zero
        result_flo = Floretion.from_string(f'0{"e" * self.flo_order}')

        for base_vec, coeff in self.base_to_nonzero_coeff.items():
            if coeff > 0:
                pass
            # reinitialize
            this_flo = Floretion(np.array([1]), np.array([base_vec]))
            center = np.array(list(this_flo.find_center_base_vectors_only(decomposition_type)))
            result_flo = result_flo + Floretion(coeff * np.ones(center.size), center, format_type="dec")

        return result_flo

    def find_center_base_vectors_only(self, decomposition_type="both"):
        """
        Find all base vectors whose product with this floretion (considered as a base vector)
        has the same sign as the product in the reverse order.

        Returns:
            A list of base vectors that commute with this floretion as a base vector.
        """
        commuting_base_vectors = set()

        # Assuming 'self' is a base vector, represented by a single nonzero coefficient
        if len(self.base_to_nonzero_coeff) != 1:
            raise ValueError("The floretion must be a single base vector")

        # Get the base vector and its order
        base_vec_self, _ = next(iter(self.base_to_nonzero_coeff.items()))

        flo_order = self.flo_order

        for base_vec in self.base_vec_dec_all:
            # Check if the signs of the products are the same
            if decomposition_type == "both":
                if (Floretion.mult_flo_sign_only(base_vec_self, base_vec, flo_order) ==
                        Floretion.mult_flo_sign_only(base_vec, base_vec_self, flo_order)):
                    commuting_base_vectors.add(base_vec)
            elif decomposition_type == "positive" or decomposition_type == "pos":
                if (Floretion.mult_flo_sign_only(base_vec_self, base_vec, flo_order) > 0 and
                        Floretion.mult_flo_sign_only(base_vec, base_vec_self, flo_order) > 0):
                    commuting_base_vectors.add(base_vec)
            elif decomposition_type == "negative" or decomposition_type == "neg":
                if (Floretion.mult_flo_sign_only(base_vec_self, base_vec, flo_order) < 0 and
                        Floretion.mult_flo_sign_only(base_vec, base_vec_self, flo_order) < 0):
                    commuting_base_vectors.add(base_vec)
            elif decomposition_type == "mixed_pos" or decomposition_type == "mixpos":
                if (Floretion.mult_flo_sign_only(base_vec_self, base_vec, flo_order) > 0 and
                        Floretion.mult_flo_sign_only(base_vec, base_vec_self, flo_order) < 0):
                    commuting_base_vectors.add(base_vec)
            elif decomposition_type == "mixed_neg" or decomposition_type == "mixneg":
                if (Floretion.mult_flo_sign_only(base_vec_self, base_vec, flo_order) > 0 and
                        Floretion.mult_flo_sign_only(base_vec, base_vec_self, flo_order) < 0):
                    commuting_base_vectors.add(base_vec)

        return commuting_base_vectors


if __name__ == "__main__":
    coeffs = [1, .5, .5, .5, .5, 1, 1, 1, 1]
    basevecs = [412472, 412722, 412171, 412711, 412777, 121111, 221222, 444244, 177777]

    coeffs = [1, .5, .5, 0.25]
    basevecs = [412, 112, 414, 111]

    yo = Floretion(coeffs, basevecs, format_type="oct")
    print(f"yo {yo.as_floretion_notation()} ")

    x = Floretion.from_string("1.0ie + 1.0ek + 1.0je")
    y = Floretion.from_string("1.0kj + 1.0ki +ee")
    z = x * y
    print(f"z {z.as_floretion_notation()} ")

    x = Floretion.from_string("1.0jj + 1.0jk + 1.0ji")
    y = Floretion.from_string("1.0ej+ii+kk")
    z = x * y
    print(f"z {z.as_floretion_notation()} ")

    x = Floretion.from_string("ii + ie + ik + ej + ki + ke + kk + ij + ek + ee + ei + kj + ji + je + jk + jj")
    y = Floretion.from_string("1.0ii+jj+kk")
    z = x * y
    print(f"z {z.as_floretion_notation()} ")

    # TEST CODE ONLY
    # glider
    # df = np.array([[1, 1247], [1, 1272], [1, 1217], [1, 1271], [1, 1277]])

    df = np.array([[1, 41247], [1, 41272], [1, 41217], [1, 41271], [1, 41277],
                   [1, 44247], [1, 44272], [1, 44217], [1, 44271], [1, 44277],
                   [1, 41444], [1, 42474], [1, 42774]])

    # df = np.array([[1, 44441]])
    # flo_x = Floretion([1, .5, -.2], [11, 21, 41],  format_type="oct")
    # flo_y = Floretion([-1], [44], format_type="oct")
    # print((flo_y).as_floretion_notation())

    flo_x = Floretion.from_string("ii +jj + ek")
    flo_y = Floretion.from_string("ie")

    commutes_ie = np.array([9, 10, 12, 15, 57, 58, 60, 63])
    flo_commutes_ie = Floretion(np.ones(commutes_ie.size), commutes_ie, format_type="dec")
    print(f"flo_commutes_ie {flo_commutes_ie.as_floretion_notation()}")

    print(flo_y.find_center())

    flo_z = flo_x * flo_y
    # print(f"flo_z {flo_z.as_floretion_notation()}")

    flo_x = Floretion.from_string("ii")
    flo_x_inv = flo_x.inverse()
    print(f"flo_x {flo_x.as_floretion_notation()} flo_x_inv {flo_x_inv.as_floretion_notation()}")

    flo_x = Floretion.from_string(".5iii +.5jjj + .5kkk")
    flo_y = Floretion.from_string("iij +  iik + jji +jjk + kki + kkj")

    flo_z = flo_x * flo_y
    # print(f"flo_z {flo_z.as_floretion_notation()}")

    flo_x = Floretion.from_string("ijk + iji + iii")
    flo_y = Floretion.from_string("iik")

    # flo_z = flo_x * flo_y
    # print(f"flo_z {flo_z.as_floretion_notation()}")

    coeffs = [1]

    flo1 = Floretion(coeffs, [10], format_type="dec")
    flo2 = Floretion(coeffs, [12], format_type="dec")

    flo3 = flo2 * flo1

    # print(f"flo1 {flo1.as_floretion_notation()}")
    # print(f"flo2 {flo2.as_floretion_notation()}")
    # print(f"flo3 {flo3.as_floretion_notation()}")

    flo4 = flo3 * flo3
    # print(f"flo4 {flo4.as_floretion_notation()}")
    # flo_c = flo_c*flo_c
    # print(flo_c.as_floretion_notation())
    # flo_t = flo_x_squared = flo_c ** 2
    # print(flo_t.as_floretion_notation())
    # flo_z = Floretion.from_string("1.0ii-3.0ij+2.0ee")

    # flo_order = 1
    # grid_flo_data = pd.read_csv(f"./data/grid.flo_{flo_order}.oct.csv", dtype={'oct': str})
    # c = Floretion.from_preloaded_data([1, 0, 0, 1], [1, 2, 4, 7], flo_order, grid_flo_data)
    # print(f"2c {c.as_floretion_notation()} ")
    # c.coeff_vec_all = Floretion.add_sp(c, c)
    # print(f"2c {c.as_floretion_notation()} ")

    # coeffs = [1]
    # coords = np.array([22, 7])
    # coords = np.array([[333, 106]])
    # coords = np.array([[355, 113]])

    # coeffs = [1]
    # coords = np.array([[355, 113]])
    # yo1 = Floretion.from_cartesian_coords(coeffs=coeffs, coords=coords)
    # coords = np.array([[333, 106]])
    # yo2 = Floretion.from_cartesian_coords(coeffs=coeffs, coords=coords)
    # z = yo1*yo2
    # print(f"yo1 {yo1.as_floretion_notation()} yo2 {yo2.as_floretion_notation()} z {z.as_floretion_notation()}")

    # flo_x = Floretion.from_string("eeeeeikkj")
    # flo_y = Floretion.from_string("eikjikeje")
    # flo_x = Floretion.from_string("eeeeeikkj")
    # flo_y = Floretion.from_string("eikjikeje")
    # flo_z = flo_x * flo_y
    # print(f"flo_x {flo_x.as_floretion_notation()} flo_y {flo_y.as_floretion_notation()} "
    #      f"flo_x*flo_y = {(flo_z).as_floretion_notation()} dec_x: {flo_x.base_to_nonzero_coeff} "
    #      f"dec_y: {flo_y.base_to_nonzero_coeff}"
    #      f"dec_z: {flo_z.base_to_nonzero_coeff} ")

    # f"grid coords {Floretion.flo_oct_to_grid(flo_z.base_to_nonzero_coeff)}" )
# flo = Floretion.from_cartesian_coords(coeffs=coeffs, coords=coords, flo_order=-1)
# print(f"flo {flo.as_floretion_notation()} ")

# c = Floretion.from_string("-2i-2.0e")
# print(c.as_floretion_notation())
# c.coeff_vec_all = Floretion.add_sp(c,c)
# print(c.as_floretion_notation())
# yo_instance.coeff_vec_all = Floretion.add_sp(yo_instance, yo_instance)
# yo_instance.coeff_vec_all = Floretion.mul_sp(yo_instance, yo_instance)
# print(f"yo {yo_instance.as_floretion_notation()} sum_of_squares {yo_instance.sum_of_squares()}")

# print(flo_z.as_floretion_notation())
# flo_z = flo_z*flo_z
# print((flo_x-flo_x).as_floretion_notation())
# print(flo_z.as_floretion_notation())
# print(flo_z.base_to_nonzero_coeff)
# print(flo_z.base_to_grid_index)

# floretion_obj.display_as_grid2()
# floretion_obj.display_as_grid()
# floretion_obj.conway(0)
# floretion_obj.conway(1000)
# df = np.array([[0.5, 137], [1, 138]])
# floretion_obj = floret_ion(df, "dec")
# print(floretion_obj.as_floretion_notation())

