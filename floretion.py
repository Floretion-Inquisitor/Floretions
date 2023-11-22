# MIT License

# Copyright (c) [2023 [Creighton Dement]

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
import cv2
import os
import re

from floboard import floboard
import floretion_base_vector


def count_bits(n):
    return bin(n).count('1')


def sgn(x):
    return -1 if x < 0 else 1


class Floretion:
    """
    Represents a Floretion, which is a type of hypercomplex number that extends the idea of quaternions.

    Attributes:
        base_to_nonzero_coeff: A dictionary mapping base vectors (in decimal) to their non-zero coefficients.
        format_type: A string representing the format type ('dec' or 'oct') of base vectors.
        max_order: An integer specifying the maximum order for the floretions.
        flo_order: An integer specifying the floretion order.
        grid_flo_loaded_data: A Pandas DataFrame holding loaded data.
        base_vec_dec_all: A NumPy array of all base vectors in decimal.
        coeff_vec_all: A NumPy array of coefficients aligned with `base_vec_dec_all`.
        base_to_grid_index: A dictionary mapping base vectors to their grid indices.
    """

    def __init__(self, coeffs_of_base_vecs, base_vecs, grid_flo_loaded_data=None, format_type="dec"):
        """
        Initializes a Floretion instance.

        Parameters:
            coeffs_of_base_vecs: A list of coefficients.
            base_vecs: A list of base vectors.
            grid_flo_loaded_data: A Pandas DataFrame of pre-loaded data. Defaults to None.
            format_type: A string indicating the format of base vectors ('dec' or 'oct'). Defaults to 'dec'.
        """
        temp_coeff_vec = np.array(coeffs_of_base_vecs)
        temp_base_vec_dec = np.array(base_vecs).astype(int)

        # Convert and store appropriate representations
        for i in range(len(base_vecs)):
            if base_vecs[i] < 0:
                coeffs_of_base_vecs[i] *= -1
                base_vecs[i] = abs(base_vecs[i])

            if format_type == "oct":
                temp_base_vec_dec[i] = int(str(base_vecs[i]), 8)

        self.base_to_nonzero_coeff = {temp_base_vec_dec[i]: coeffs_of_base_vecs[i] for i in
                                      range(len(temp_base_vec_dec)) if
                                      abs(coeffs_of_base_vecs[i]) > np.finfo(float).eps}

        self.format_type = format_type

        self.max_order = 10  # Define an appropriate value for max_order
        self.flo_order = self.find_flo_order(temp_base_vec_dec, self.max_order)

        # Load the complete listing of base vectors
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, 'Floretion/data', f"grid.flo_{self.flo_order}.oct.csv")

        if grid_flo_loaded_data is None:
            self.grid_flo_loaded_data = pd.read_csv(file_path, dtype={'oct': str})
        else:
            self.grid_flo_loaded_data = grid_flo_loaded_data

        self.base_vec_dec_all = self.grid_flo_loaded_data['floretion'].to_numpy()
        self.coeff_vec_all = np.zeros_like(self.base_vec_dec_all, dtype=float)

        # Populate coefficient array based on provided coeffs and base_vecs
        for coeff, base_vec in zip(temp_coeff_vec, temp_base_vec_dec):
            idx = np.where(self.base_vec_dec_all == base_vec)[0]
            self.coeff_vec_all[idx] = coeff

        self.base_to_grid_index = {}
        for base_vec, _ in self.base_to_nonzero_coeff.items():
            index_row = self.grid_flo_loaded_data[self.grid_flo_loaded_data['floretion'] == base_vec].index[0]
            self.base_to_grid_index[base_vec] = index_row

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

    @classmethod
    def from_cartesian_coords(cls, coeffs, coords, flo_order=-1):
        """
           Creates a Floretion object from cartesian coordinates.

           Parameters
           ----------
           coeffs : list
               List of coefficients.
           coords : list of tuple
               List of tuples representing cartesian coordinates.
           flo_order : int, optional
               The order of the floretion. Defaults to -1, which means the method will try to determine it.

           Returns
           -------
           Floretion
               A Floretion object.
           """
        if len(coeffs) != len(coords):
            raise ValueError("The number of coefficients must match the number of coordinates.")

        max_coord_value = int(np.sqrt(4 ** flo_order) - 1)

        if flo_order == -1:
            max_coord_value = np.max(np.array(coords))
            flo_order = 0
            while True:
                if max_coord_value < np.sqrt(4 ** flo_order) - 1:
                    break
                flo_order += 1

        for x, y in coords:
            if x > max_coord_value or y > max_coord_value:
                raise ValueError(
                    f"Coordinates x={x} and y={y} exceed the maximum allowable value of {max_coord_value} for flo_order {flo_order}.")

        instance = cls.__new__(cls)
        instance.flo_order = flo_order

        # Load the data from a file path generated internally
        file_path = f"./data/grid.flo_{flo_order}.oct.csv"
        instance.grid_flo_loaded_data = pd.read_csv(file_path, dtype={'oct': str})

        instance.base_vec_dec_all = instance.grid_flo_loaded_data['floretion'].to_numpy()
        instance.coeff_vec_all = np.zeros_like(instance.base_vec_dec_all, dtype=float)

        # Initialize base_to_nonzero_coeff and base_to_grid_index as empty dictionaries
        instance.base_to_nonzero_coeff = {}
        instance.base_to_grid_index = {}

        for coeff, (x, y) in zip(coeffs, coords):
            index_row = instance.grid_flo_loaded_data[
                (instance.grid_flo_loaded_data['x_coord'] == x) & (
                        instance.grid_flo_loaded_data['y_coord'] == y)].index.values[0]
            floretion_val = instance.grid_flo_loaded_data.at[index_row, 'floretion']

            # Update coefficient vector and base_to_nonzero_coeff
            instance.coeff_vec_all[index_row] = coeff
            instance.base_to_nonzero_coeff[floretion_val] = coeff

            # Update base_to_grid_index
            instance.base_to_grid_index[floretion_val] = index_row

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
        oct111  = int('1' * flo_order, 8)

        pre_sign = sgn(a_base_val) * sgn(b_base_val)
        a_base_val = abs(a_base_val)
        b_base_val = abs(b_base_val)

        # Shift every 3-bits of "a" one to the left
        a_cyc = ((a_base_val << 1) & oct666) | ((a_base_val >> 2) & oct111)

        cyc_sign = 1 if count_bits((a_cyc & b_base_val) & bitmask) & 0b1 else -1
        ord_sign = 1 if count_bits(bitmask) & 0b1 else -1

        return (pre_sign * cyc_sign * ord_sign)

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

    def __mul__(self, other):
        """
        Overloads the multiplication (*) operator for Floretion instances.

        Parameters:
            other: A Floretion instance or a scalar (int, float) for scalar multiplication.

        Returns:
            A new Floretion instance resulting from the multiplication.

        Examples:
            f1 = Floretion([1, 2], [3, 4])
            f2 = Floretion([1, 2], [3, 4])
            f3 = f1 * f2  # Floretion multiplication
            f4 = f1 * 2   # Scalar multiplication
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

    @staticmethod
    def flo_oct_to_grid(base_dec):
        flo_octal = oct(base_dec)[2:]
        # print(f"{flo_octal}{flo_octal}")
        x_coord = 0
        y_coord = 0

        flo_order = len(flo_octal)
        num_rows_or_num_cols_in_grid = 2 ** flo_order
        shift_amount = num_rows_or_num_cols_in_grid / 2

        for i, digit in enumerate(flo_octal):
            digit = int(digit)  # Converti il carattere in un intero
            # print(f"{digit}")

            if digit == 1:
                pass
            elif digit == 2:
                y_coord += shift_amount
            elif digit == 4:
                x_coord += shift_amount
                y_coord += shift_amount
            elif digit == 7:
                x_coord += shift_amount

            # print(
            #    f"flo_order {flo_order}, digit {digit}, shift_amount {shift_amount}, x_coord {x_coord}, y_coord {y_coord}")
            shift_amount /= 2

        x_coord = int(x_coord)
        y_coord = int(y_coord)
        return np.array([x_coord, y_coord])

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

    def sum_of_squares(self):
        return sum(coeff ** 2 for coeff in self.coeff_vec_all)

    def abs(self):
        return np.sqrt(self.sum_of_squares())

    @classmethod
    def from_string(cls, flo_string, format_type="dec"):
        # Error check 1: No invalid characters
        if not all(c in "0123456789ijke.+ -" for c in flo_string):
            raise ValueError("Invalid character in floretion string.")

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

    @staticmethod
    def grid_to_flo_oct(x_coord, y_coord, num_iter):
        x_sum = 0
        y_sum = 0
        result = 0

        for i in range(num_iter):
            base = 8
            p_const = 2 ** (num_iter - 1 - i)

            if x_coord <= (p_const + x_sum) and y_coord <= (p_const + y_sum):
                result += 1 * (base ** i)  # Lower-left quadrant

            elif x_coord <= (p_const + x_sum) and y_coord > (p_const + y_sum):
                x_sum += 0
                y_sum += p_const
                result += 2 * pow(base, i)

            elif x_coord > (p_const + x_sum) and y_coord > (p_const + y_sum):
                x_sum += p_const
                y_sum += p_const
                result += 4 * (base ** i)  # upper-right quadrant

            elif x_coord > (p_const + x_sum) and y_coord <= (p_const + y_sum):
                x_sum += p_const
                y_sum += 0
                result += 7 * (base ** i)  # Upper-right quadrant

        return result

    def display_as_grid2(self):
        board_size = 2 ** self.flo_order
        chessboard = floboard(board_size)
        for base_dec, value in self.base_to_nonzero_coeff.items():
            # print(f"{base_dec}{base_dec}, {value}{value}")
            coords = self.flo_oct_to_grid(base_dec)
            print(f"display_as_grid coords: ({coords[0]}, {coords[1]})")
            chessboard.update_square(coords[1], coords[0], value)
            # chessboard.update_square(2, 0, 1)
        chessboard.display()

    def display_as_grid(self, resize_size=(512, 512)):
        board_size = 2 ** self.flo_order
        board_image = np.zeros((board_size, board_size), dtype=np.uint8)  # Black 2D array

        # for row in self.df:  # Assuming df is a NumPy array with the coefficients and base_dec
        #    coefficient, base_dec = row
        for base_dec, value in self.base_to_nonzero_coeff.items():
            x_coord, y_coord = self.flo_oct_to_grid(base_dec)  # Get the coordinates using your existing function
            print(f"x_coord {x_coord}, y_coord {y_coord}")
            # x_coord = 0
            # y_coord = 7
            board_image[board_size - 1 - y_coord, x_coord] = value  # Set the value in the corresponding cell

        board_image = 255 * board_image
        board_image = cv2.applyColorMap(board_image, cv2.COLORMAP_OCEAN)

        board_image_resized_up = cv2.resize(board_image, resize_size, interpolation=cv2.INTER_LINEAR)

        # Add grid lines
        grid_size = 512 // board_size
        for i in range(0, 513, grid_size):
            cv2.line(board_image_resized_up, (i, 0), (i, 512), (0, 155, 0), 1)  # vertical lines
            cv2.line(board_image_resized_up, (0, i), (512, i), (0, 155, 0), 1)  # horizontal lines

        cv2.imshow('Floretion Board', board_image_resized_up)
        cv2.imwrite('floretion_board_with_grid.png', board_image_resized_up)  # Save the image
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def next_step(self, grid):
        # Create a copy of the grid to avoid conflicts during updating
        new_grid = grid.copy()

        grid_size_x = len(grid)
        grid_size_y = len(grid[0])

        # Loop through each cell and apply the Game of Life rules
        for i in range(grid_size_x):
            for j in range(grid_size_y):
                alive_neighbors = sum(
                    [grid[x % grid_size_x][y % grid_size_y] for x, y in self.neighbors(i, j)]
                )
                if grid[i][j] == 1:  # If the cell is alive
                    if alive_neighbors < 2 or alive_neighbors > 3:
                        new_grid[i][j] = 0  # Dies
                else:  # If the cell is dead
                    if alive_neighbors == 3:
                        new_grid[i][j] = 1  # Becomes alive

        return new_grid

    def neighbors(self, x, y):
        # Restituisce le coordinate dei vicini di una cella data
        return [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1), (x, y - 1), (x, y + 1), (x + 1, y - 1), (x + 1, y),
                (x + 1, y + 1)]

    def grid_to_coordinates(self, grid):
        # Converte la griglia in un array di coordinate
        coords = np.array([[i, j] for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j] == 1])
        return coords



    def inverse(self):
        # Check if flo is a single base floretion (positive or negative)
        non_zero_elements = [coeff for coeff in self.coeff_vec_all if coeff != 0]
        if len(non_zero_elements) == 1 and abs(non_zero_elements[0]) == 1:
            # flo is a base floretion
            if self * self == Floretion.from_string("e"):
                return self  # Inverse is the floretion itself
            else:
                return -1*self  # Inverse is the negative of the floretion
        else:
            raise ValueError("Provided floretion is not a base floretion.")

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


    def conway(self, n):
        # Create a directory to store the CSV files if it doesn't exist
        output_dir = "data/conway"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize the grid like you did before
        board_size = 2 ** self.flo_order
        grid = np.zeros((board_size, board_size), dtype=int)

        for base_dec, value in self.flomap_base_to_nonzero_coeff.items():
            coords = self.flo_oct_to_grid(base_dec)
            x, y = coords
            grid[x][y] = value

        # Simulate n steps of the game
        for i in range(n + 1):  # +1 to include the 0th step
            # Get the live coordinates
            live_coords = self.grid_to_coordinates(grid)

            # Prepare data as a pandas DataFrame
            data = []
            for x, y in live_coords:
                base_dec = self.grid_to_flo_oct(x, y, self.flo_order)
                data.append([base_dec, x, y])

            df = pd.DataFrame(data, columns=["floretion", "x_coord", "y_coord"])

            # Save DataFrame to CSV
            number_str = str(i).zfill(4)  # Making sure it has 4 digits

            csv_file_path = os.path.join(output_dir, f"conway.flo.{number_str}.csv")
            df.to_csv(csv_file_path, index=False)

            print(f"Conway iteration number {i}: Saved to {csv_file_path}")

            # Update grid for the next iteration
            grid = self.next_step(grid)


def draw_triangle(image, vertex1, vertex2, vertex3, color=(255, 255, 255)):
    cv2.line(image, tuple(vertex1), tuple(vertex2), color, 1)
    cv2.line(image, tuple(vertex2), tuple(vertex3), color, 1)
    cv2.line(image, tuple(vertex3), tuple(vertex1), color, 1)


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
    flo_commutes_ie = Floretion(np.ones(commutes_ie.size), commutes_ie ,  format_type="dec")
    print(f"flo_commutes_ie {flo_commutes_ie.as_floretion_notation()}")


    print(flo_y.find_center())

    flo_z = flo_x * flo_y
    #print(f"flo_z {flo_z.as_floretion_notation()}")

    flo_x = Floretion.from_string("ii")
    flo_x_inv =  flo_x.inverse()
    print(f"flo_x {flo_x.as_floretion_notation()} flo_x_inv {flo_x_inv.as_floretion_notation()}")

    flo_x = Floretion.from_string(".5iii +.5jjj + .5kkk")
    flo_y = Floretion.from_string("iij +  iik + jji +jjk + kki + kkj")

    flo_z = flo_x * flo_y
    #print(f"flo_z {flo_z.as_floretion_notation()}")

    flo_x = Floretion.from_string("ijk + iji + iii")
    flo_y = Floretion.from_string("iik")

    #flo_z = flo_x * flo_y
    #print(f"flo_z {flo_z.as_floretion_notation()}")

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
