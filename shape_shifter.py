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

import cv2
import numpy as np
import os
import re
from floretion import Floretion
from SierpinskiFlo import SierpinskiFlo




#def six_ellipses(floretion, iteration_int, order):
    # X = .5(i + j + k + e)
    # Y = e(e is unit)
#    Floretion

    # Define X(i) as the coefficient of i in X, X(j) as the coefficient of j in X, etc.

    # Z = X*Y # quaternion multiplication (i*j = k, j*i = -k, i*e = i, etc)
    #
    # if n mod 3 == 0:
    #   Z(j) = Z(j) - (floor(Z(i)) - Z(i))/2
    #
    # else if n mod 3 == 1:
    #   Z(k) = Z(k) - (floor(Z(j)) - Z(j))/2
    #
    # else if n mod 3 == 2:
    #   Z(i) = Z(i) - (floor(Z(k)) - Z(k))/2
    #
    #  n = n + 1
    #
    #  Y = Z
    #
    #  result.add(Y(e))
    #
    #  goto 0


def parse_special_commands(input_str, order):
    """Parse and interpret special commands like Cp(), Cn(), and Cb() with order validation."""
    command_match = re.match(r"(Cp|Cn|Cb)\(([\w+.-]+)\)", input_str)
    if command_match:
        command, base_vec = command_match.groups()

        # Check that base_vec is the correct length and only contains valid characters
        valid_chars = "0123456789ijke.+ -"
        if len(base_vec) != order or not all(c in valid_chars for c in base_vec):
            raise ValueError(f"Invalid base vector length or character(s). Expected length {order}.")

        floretion_base_vec = Floretion.from_string(base_vec)
        base_vec_oct_str = base_vec.replace("i","1").replace("j", "2").replace("k", "4").replace("e", "7")

        base_vec_oct_str = str(int(base_vec_oct_str, 8))
        #print(base_vec_oct_str)
        storage_type = "json"
        if command == "Cp":
            centers_data = Floretion.load_centers(floretion_base_vec.flo_order, decomposition_type="pos", storage_type=storage_type)
        elif command == "Cn":
            centers_data = Floretion.load_centers(floretion_base_vec.flo_order, decomposition_type="neg", storage_type=storage_type)
        elif command == "Cb":
            centers_data = Floretion.load_centers(floretion_base_vec.flo_order, decomposition_type="both", storage_type=storage_type)
        else:
            raise ValueError("Unknown command type. Use Cp, Cn, or Cb.")
        #print(centers_data)
        coeff_array = np.ones(len(centers_data))
        return Floretion(coeff_array, np.array(centers_data[base_vec_oct_str]))
    else:
        # Check if input contains valid characters
        if not all(c in "0123456789ijke.+ -" for c in input_str):
            raise ValueError("Invalid character in floretion string.")
        return Floretion.from_string(input_str)

def get_base_vec_centroid_dist(base_vector):
        """
        For each base vector of a given order, returns the coordinates of the center (centroid) of the
        equilateral triangle associated with it, along with the final distance, and color for further processing.

        Args:
            base_vector (str): The base vector in octal representation.

        Returns:
            tuple:  final distance
        """
        x, y = 0, 0

        distance = 1# self.distance_scale_fac
        sign_distance = -1

        for digit in base_vector:
            # digit 4 should be at 330 and digit 1 at 210, i.e. 1 and 4 should be reversed,
            # but writing the other away around here prevents us from having to call flip
            if digit == '7':
                sign_distance *= -1
            else:
                if digit == '4':
                    angle = 210

                elif digit == '2':
                    angle = 90

                elif digit == '1':
                    angle = 330
                else:
                    print(f"Invalid digit {digit}")
                    return

                x += np.cos(np.radians(angle)) * distance * sign_distance
                y += np.sin(np.radians(angle)) * distance * sign_distance




            distance /= 2  # Halve the distance for the next iteration



        centroid_distance = np.sqrt(x ** 2 + y ** 2)

        return centroid_distance


def get_basevec_coords(base_vector):
    """
    For each base vector of a given order, returns the coordinates of the center (centroid) of the
    equilateral triangle associated with it, along with the final distance, and color for further processing.

    Args:
        base_vector (str): The base vector in octal representation.

    Returns:
        tuple:  final distance
    """
    x, y = 0, 0

    distance = 1  # self.distance_scale_fac
    sign_distance = -1

    for digit in base_vector:
        # digit 4 should be at 330 and digit 1 at 210, i.e. 1 and 4 should be reversed,
        # but writing the other away around here prevents us from having to call flip
        if digit == '7':
            sign_distance *= -1
        else:
            if digit == '4':
                angle = 210

            elif digit == '2':
                angle = 90

            elif digit == '1':
                angle = 330
            else:
                print(f"Invalid digit {digit}")
                return

            x += np.cos(np.radians(angle)) * distance * sign_distance
            y += np.sin(np.radians(angle)) * distance * sign_distance

        distance /= 2  # Halve the distance for the next iteration


    return [x, y]

def clip_coeffs(floretion, clip_threshold):

    coeff_array = []
    #print(floretion.coeff_vec_all)

    for base_vector, coeff in zip(floretion.grid_flo_loaded_data['oct'], floretion.coeff_vec_all):

        distance_to_unit = get_base_vec_centroid_dist(base_vector)


        if distance_to_unit  >  clip_threshold:
            this_coeff = 0.0
            coeff_array.append(0.0)
        else:
            print(f"coeff {coeff}  distance to unit {distance_to_unit} clip_threshold {clip_threshold}")
            coeff_array.append(coeff)

    coeff_array_final = np.array(coeff_array)
    #print(coeff_array_final)

    return Floretion(coeff_array_final, floretion.base_vec_dec_all, floretion.grid_flo_loaded_data)


def ball_coeffs(floretion, scale_coeff=1.0, chosen_orientation = "both"):
    '''
    '''

    coeff_array = []

    for base_vector, coeff in zip(floretion.grid_flo_loaded_data['oct'], floretion.coeff_vec_all):

        basevec_coords = get_basevec_coords(str(base_vector))

        orientation = SierpinskiFlo.calculate_orientation(base_vector)


        coeff_array.append(coeff * basevec_coords[0])

        final_coeff = 0

        #if orientation == "up":
        #    coeff_array.append(coeff*scale_coeff*basevec_coords[0])
        #else:
        #    coeff_array.append(coeff*scale_coeff*basevec_coords[1])

    coeff_array_final = coeff_array

    return Floretion(coeff_array_final, floretion.base_vec_dec_all, floretion.grid_flo_loaded_data)



if __name__ == "__main__":

    height, width = 2048, 2048

    gamma = np.linspace(0, 2, num=100)

    order = 6
    img = np.zeros((height, width, 3), np.uint8)
    sierp = Floretion.get_typical_floretions("uniform_flo", order)


    center_flo = parse_special_commands("Cn(iiiiii)", 6)

    flo_clipped = clip_coeffs(sierp, .75)
    flo_clipped  = flo_clipped  * flo_clipped
    flo_clipped = flo_clipped * flo_clipped
    print(flo_clipped.as_floretion_notation())


    #floretion_instance = ball_coeffs(Floretion.from_string("eeeee"), scale_coeff=1)
    flo_clipped = Floretion.normalize_coeffs(flo_clipped, 2)
    SierpinskiFlo(flo_clipped , img, plot_type='triangle').plot_floretion()


    #img = cv2.flip(img, 1)

    #dim = (516, 516)
    #img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


    #cv2.imshow('SierpinskiFlo', img)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #exit(-1)

    # create global?
    gam_index = 0
    #floretion_instance = center_flo
    # 1: np.sqrt((1/3))
    # 2: np.sqrt((1/9))
    #order = 4
    #sierp_flo_fis = Floretion.get_typical_floretions("sierp_flo", order )
    #sierp_flo = sierp_flo_fis * sierp_flo_fis
    #print(f"x*x={sierp_flo.as_floretion_notation()}")

    #sierp_flo = sierp_flo * sierp_flo_fis
    #print(f"x*x*x={sierp_flo.as_floretion_notation()}")

    #norm_fac = np.sqrt(1. / (order + 1) ** 3)
    #exit(-1)
    #axis_flo = get_typical_floretions("axis_ijk", center_flo.flo_order)

    #frac_flo = Floretion.from_string("eeeeee")
    #floretion_instance = axis_flo
    #gamma = [1]
    for gam in gamma:
        print(f"Gamma index {gam_index}, value {gam}")

        floretion_instance = ball_coeffs(center_flo, scale_coeff=gam)

        floretion_instance = Floretion.normalize_coeffs(floretion_instance*floretion_instance)
        #print(floretion_instance.as_floretion_notation())
        sierp_flo = sierp_flo*sierp_flo
        print(f"x*y={sierp_flo.as_floretion_notation()}")
        #floretion_instance = Floretion.mirror(floretion_instance, axis="I") + Floretion.mirror(floretion_instance, axis="J") + Floretion.mirror(
        #    floretion_instance, axis="K")
        #floretion_instance = (1/3)*floretion_instance
    #print(f"x*y={floretion_instance.as_floretion_notation()}")

        final_scale_facs = np.linspace(2, 4, num=1)
        final_scale_index = 0
        # Create SierpinskiFlo instance
        for final_scale_fac in final_scale_facs:
            floretion_show = Floretion.normalize_coeffs(floretion_instance, final_scale_fac)
            floretion_show = clip_coeffs(floretion_show, 2)
            sierpinski_instance = SierpinskiFlo(floretion_show , img, plot_type='triangle')
            sierpinski_instance.plot_floretion()
            filedir = f"./data/base_folder/flo_pet/order_{floretion_instance.flo_order}"
            if not os.path.exists(filedir):
                os.makedirs(filedir)

            gam_index_formatted = "{:05d}".format(gam_index)
            scalefac_formatted = "{:05d}".format(final_scale_index)
            filename = filedir + f"/gam_index_{gam_index_formatted}.scale_fac_{scalefac_formatted }.png"

            final_scale_index += 1
            cv2.imwrite(filename, img)

        gam_index += 1
        #exit(-1)




    #cv2.imwrite(filename, img)

    #img = cv2.flip(img, 1)
    #cv2.imshow('SierpinskiFlo', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

