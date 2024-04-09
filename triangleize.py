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
import colorsys
from floretion import Floretion


class Triangleize:
    """
    A class to represent the Sierpinski structure based on floretion mathematics.

    Attributes:
        floretion (Floretion): An instance of the Floretion class.
        base_vectors (list): A list of octal strings representing base vectors.
        coeffs (list): A list of coefficients for each base vector.
        img (ndarray): The image to be plotted on.
        plot_type (str): The type of plot ('dot' or 'triangle').
        height (int): The height of the image.
        width (int): The width of the image.
        distance_scale_fac (int): The scaling factor for distances in the image.
    """

    def __init__(self, floretion_object, image, plot_type='dot', distance_scale_fac=4):
        """
               Initializes an instance of the SierpinskiFlo class.

               Args:
                   floretion_object (Floretion): An instance of the Floretion class.
                   image (ndarray): The image to be plotted on.
                   plot_type (str, optional): The type of plot ('dot' or 'triangle'). Defaults to 'dot'.
                   distance_scale_fac (int, optional): The scaling factor for distances in the image. Defaults to 4.
        """
        self.floretion = floretion_object
        self.flo_order = self.floretion.flo_order
        self.base_vectors = self.floretion.grid_flo_loaded_data['oct']
        self.coeffs = self.floretion.coeff_vec_all
        self.max_coeff = np.absolute(self.coeffs).max()
        self.img = image
        self.plot_type = plot_type
        self.height, self.width = self.img.shape[0], self.img.shape[1]
        self.distance_scale_fac = distance_scale_fac

    def draw_dot(self, x, y, brightness, color):
        """
          Draws a dot at the specified coordinates with the given brightness and color.

          Args:
              x (int): The x-coordinate.
              y (int): The y-coordinate.
              brightness (float): The brightness level [0, 1].
              color (tuple): The BGR color.
        """

        # Scale each component of the color
        scaled_color = tuple(int(c * brightness) for c in color)

        # Radius is a placeholder; you can change this
        radius = 1

        # Draw the dot using OpenCV's circle function
        cv2.circle(self.img, (int(x), int(y)), radius, scaled_color, -1)

    def draw_triangle(self, x, y, height, orientation, color):
        """
          Draws a dot at the specified coordinates with the given brightness and color.

          Args:
              x (int): The x-coordinate.
              y (int): The y-coordinate.
              height: height of equilateral triangle
              orientation: whether to draw triangle facing upwards or downwards
              color (tuple): The BGR color.
          """
        # calculate the vertices of the equilateral triangle
        half_base = np.sin(np.pi / 3) * height
        if orientation == 'up':
            vertices = np.array([[x, y - height], [x - half_base, y + height / 2], [x + half_base, y + height / 2]],
                                np.int32)
        else:  # down
            vertices = np.array([[x, y + height], [x - half_base, y - height / 2], [x + half_base, y - height / 2]],
                                np.int32)
        vertices = vertices.reshape((-1, 1, 2))

        # color and brightness are placeholders; you can replace these based on your needs
        # color = (255, 255, 255)  # White
        # tuple(int(c * brightness) for c in color)

        # color_with_alpha = tuple(color[0], color[1], color[2], 255)
        # cv2.circle(self.img, (int(x), int(y)), int(r), color_with_alpha, -1)
        color = (int(color[0]), int(color[1]), int(color[2]))
        # print(color)
        cv2.fillConvexPoly(self.img, vertices, color)

    @staticmethod
    def calculate_orientation(base_vector):
        """
        Calculates the orientation of a triangle based on its associated base vector.

        Args:
            base_vector (str): The base vector of the floretion in octal representation.

        Returns:
            str: The orientation ('up' or 'down').
        """
        octal_str = base_vector
        count = 0
        # Instead of counting parity of the digits below, we could test if (base_vector)^2 = +1 or -1,
        # which seems more elegant mathematically but is probably not as fast computationally in current code
        for digit in octal_str:
            if digit in '124':
                count += 1

        # Determine the order of the floretion
        order = len(octal_str)

        # Check the orientation based on the count and the order
        if count % 2 == 0:  # x^2 = 1
            orientation = 'up' if order % 2 == 0 else 'down'
        else:  # x^2 = -1
            orientation = 'down' if order % 2 == 0 else 'up'

        return orientation

    def plot_floretion(self, title=None, highlight_base_vec=None):
        """
        Plots the floretion using either dots or triangles based on the plot type.
        """

        # Add the title to the image if provided
        if title:
            cv2.putText(self.img, title, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        for base_vector, coeff in zip(self.base_vectors, self.floretion.coeff_vec_all):
            x, y, final_distance, color = self.place_base_vecs(str(base_vector), coeff)
            brightness = abs(coeff)
            abs_mean = 2.0 * abs(np.mean(coeff))
            if coeff > 0:
                pass
                # color[0] = abs_mean*color[2]

                # color[1] = 0
                # color[2] = 255
                # place_holder = color[0]
                # color[0] = color[1]
                # color[1] = color[2]
                # color[2] = place_holder
            else:
                # color[0] = 255
                # color[1] = 0

                # color[2] = abs_mean*color[2]
                pass
                # color[2] = 0
                # color[1] = 0
                # color[2] = 0

            if brightness < .1:
                brightness = .1

            orientation = self.calculate_orientation(base_vector)
            # print(f" base_vector = {base_vector}, highlight_base_vec = {highlight_base_vec}")

            # If the current base vector is the one to highlight, set its color to white
            is_highlight_base_vec = False
            if base_vector == highlight_base_vec:
                is_highlight_base_vec = True
                # print(f" Full brightness for base_vector = {base_vector} = highlight_base_vec = {highlight_base_vec}")
                color = (255, 255, 255)  # White color
                brightness = 1  # Full brightness

            if self.plot_type == 'dot' or self.plot_type == 'dots':
                self.draw_dot(x, y, brightness, color)
            elif self.plot_type == 'triangle' or self.plot_type == 'triangles':
                self.draw_triangle(x, y, final_distance, orientation, color)
                # Add the octal representation on top of the triangle
                # cv2.putText(self.img, str(base_vector), (int(x)-20, int(y)+0), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 2)
                if is_highlight_base_vec:
                    pass
                    # cv2.putText(self.img, str(base_vector), (int(x) - 10, int(y) + 0), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1)
            # cv2.putText(self.img, str(base_vector), (int(x)-30, int(y)+5), cv2.FONT_HERSHEY_PLAIN, .5, (255, 255, 255), 2)
            else:
                print(f"Unknown plottype {self.plot_type}")

    # ax.text(x, y, label, fontsize=12)

    def place_base_vecs(self, base_vector, coeff):
        x, y = self.height // 2, self.width // 2
        y_offset = int(self.height * 0.1)
        y += y_offset

        distance = self.height // self.distance_scale_fac
        sign_distance = -1

        # Initialize base hue, and set increments for hue adjustment

        color_hue = []
        color_sat = []

        for digit in base_vector:
            # digit 4 should be at 330 and digit 1 at 210, i.e. 1 and 4 should be reversed,
            # but writing the other away around here prevents us from having to call flip
            if digit == '7':
                sign_distance *= -1
                color_hue.append(np.nan)
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

                if sign_distance == -1:
                    color_hue.append(360 - angle)
                else:
                    color_hue.append(360 - angle)

                x += np.cos(np.radians(angle)) * distance * sign_distance
                y += np.sin(np.radians(angle)) * distance * sign_distance

            distance /= 2
            color_sat.append(distance)
        # Saturation and Brightness (Value)
        changed_dir_count = np.count_nonzero(np.isnan(color_hue))
        if changed_dir_count > 15:
            saturation = abs(np.log(changed_dir_count))
        else:
            saturation = 1.0

        if self.max_coeff > 0:
            brightness = abs(coeff) / self.max_coeff
        else:
            brightness = 0

        processed_hue = np.array(color_hue)
        processed_sat = np.array(color_sat)
        max_sat = processed_sat.max()
        processed_sat = processed_sat / max_sat
        processed_hue = np.nansum(processed_hue * processed_sat)
        # print(f'color hue {np.array(color_hue)}')
        # print(f'color sat {np.array(color_sat)}')

        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(processed_hue, saturation, brightness)
        color_array = (np.array(rgb) * 255).astype(int)
        # print(color_array)

        return x, y, 1.8 * distance, color_array


def decimal_to_octal(decimal):
    return format(int(decimal), 'o')


if __name__ == "__main__":

    # print(colorsys.rgb_to_hsv(0, 0, 255))
    # exit(-1)
    height, width = 4 * 1024, 4 * 1024

    flo_order = 6
    zero_flo = Floretion.from_string(f'0{"e" * flo_order}')
    unit_flo = Floretion.from_string(f'1{"e" * flo_order}')

    new_coeffs_sierp = []
    new_coeffs_sierp_i = []
    new_coeffs_sierp_j = []
    new_coeffs_sierp_k = []

    for base in zero_flo.base_vec_dec_all:

        print(f'dto {decimal_to_octal(base)}')
        if '7' in decimal_to_octal(base):
            new_coeffs_sierp.append(0.0)
        else:
            new_coeffs_sierp.append(1.0)

        if '1' in decimal_to_octal(base):
            new_coeffs_sierp_i.append(0.0)
        else:
            new_coeffs_sierp_i.append(1.0)

        if '2' in decimal_to_octal(base):
            new_coeffs_sierp_j.append(0.0)
        else:
            new_coeffs_sierp_j.append(1.0)

        if '4' in decimal_to_octal(base):
            new_coeffs_sierp_k.append(0.0)
        else:
            new_coeffs_sierp_k.append(1.0)

    new_coeffs_sierp = np.array(new_coeffs_sierp)
    new_coeffs_sierp_i = np.array(new_coeffs_sierp_i)
    new_coeffs_sierp_j = np.array(new_coeffs_sierp_j)
    new_coeffs_sierp_k = np.array(new_coeffs_sierp_k)

    norm_fac = (np.sqrt((1 / 3)) ** (flo_order))
    sierp_flo = Floretion(norm_fac * new_coeffs_sierp, zero_flo.base_vec_dec_all, format_type="dec")
    sierp_flo_i = Floretion(norm_fac * new_coeffs_sierp_i, zero_flo.base_vec_dec_all, format_type="dec")
    sierp_flo_j = Floretion(norm_fac * new_coeffs_sierp_j, zero_flo.base_vec_dec_all, format_type="dec")
    sierp_flo_k = Floretion(norm_fac * new_coeffs_sierp_k, zero_flo.base_vec_dec_all, format_type="dec")

    # Initialize VideoWriter
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # FourCC code for MP4
    # out = cv2.VideoWriter('sierpinski_animation.mp4', fourcc, 30, (width, height))

    # out.write(img_t)

    # Release the VideoWriter
    # out.release()

    # Optional: Release any additional OpenCV objects
    # cv2.destroyAllWindows()

    # floretion_instance = Floretion.from_string(".2ii+ij+ik+ie+ei+ej+ek+ee+ji+jj+jk+je+ki+kj+kk+ke")



    centers_data_pos = Floretion.load_centers(flo_order, decomposition_type="pos")
    centers_data_neg = Floretion.load_centers(flo_order, decomposition_type="neg")
    grid_flo_loaded_data_all = zero_flo.grid_flo_loaded_data

    filedir = f"./data/triangleize/fisch_time{flo_order}"
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    flo_index = 0
    flo_to_map = sierp_flo
    for flo in centers_data_neg.keys():

        coeff_array = np.ones(len(centers_data_pos))
        flo_to_pos = Floretion(coeff_array, np.array(centers_data_pos[flo]))
        flo_to_neg = Floretion(coeff_array, np.array(centers_data_neg[flo]))
        img = np.zeros((height, width, 3), np.uint8)

        flo_to_map = flo_to_pos * flo_to_map * flo_to_neg

        if flo_index % 3 == 0:
            flo_to_map = flo_to_map * sierp_flo_i
        elif flo_index % 3 == 1:
            flo_to_map = flo_to_map * sierp_flo_j
        elif flo_index % 3 == 2:
            flo_to_map = flo_to_map * sierp_flo_k

        #flo_to_map = flo_to_map * flo_to_map

        flo_to_map = Floretion.normalize_coeffs(flo_to_map, 4)
        print(flo_to_map.as_floretion_notation())
        floA = Triangleize(flo_to_map, img, plot_type='triangle')
        floA.plot_floretion()
        flo_index += 1

        filename = filedir + f"/{format(int(flo), 'o')}.png"
        cv2.imwrite(filename, img)

    exit(-1)
    tes_index = 0
    for tes_scalar in np.arange(0, 2, .1):
        coeff_array[-10:-1] = 1
        print(coeff_array)

        print(flo_to_pos.as_floretion_notation())

        # flo_to_neg = Floretion(np.ones(len(centers_data_neg[flo])), np.array(centers_data_neg[flo]))
        # flo_to_map = flo_to_pos*flo_to_neg

        # identities
        # sierp_flo_i * sierp_flo_k = j
        # sierp_flo_j * sierp_flo_i = k
        # sierp_flo_k * sierp_flo_j = i
        flo_to_map = flo_to_pos
        # flo_to_map = flo_to_map*flo_to_map
        print(flo_to_map.as_floretion_notation())

        floA = Triangleize(flo_to_map, img, plot_type='triangle')
        floA.plot_floretion()
        filename = filedir + f"/{format(int(flo), 'o')}.tes_{'{:05.0f}'.format(tes_index)}.png"
        tes_index += 1
        cv2.imwrite(filename, img)

    flo_index += 1

# imgA = np.zeros((height, width, 3), np.uint8)
# floA = Floretion.normalize_coeffs(this_flo, 1)


# cv2.waitKey(0)
# cv2.destroyAllWindows()
