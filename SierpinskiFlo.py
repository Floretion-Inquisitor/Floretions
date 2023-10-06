import cv2
import numpy as np
import os
from floretion import Floretion


def gauss_coeffs(time, mean, var):
    return np.exp(-0.5 * ((time - mean) ** 2) / var)


def generate_coeffs(time, num_coeffs, mean_var_list):
    coeffs = np.zeros(num_coeffs)
    for i in range(num_coeffs):
        mean, var = mean_var_list[i]
        coeffs[i] = gauss_coeffs(time, mean, var)
    return coeffs


class SierpinskiFlo:
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
        self.base_vectors = self.floretion.grid_flo_loaded_data['oct']
        self.coeffs = self.floretion.coeff_vec_all

        self.img = image
        self.plot_type = plot_type
        self.height, self.width = img.shape[0], img.shape[1]
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

    def draw_triangle(self, x, y, height, orientation, brightness, color):
        """
          Draws a dot at the specified coordinates with the given brightness and color.

          Args:
              x (int): The x-coordinate.
              y (int): The y-coordinate.
              height: height of equilateral triangle
              orientation: whether to draw triangle facing upwards or downwards
              brightness (float): The brightness level [0, 1].
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

        scaled_color = tuple(int(c * brightness) for c in color)

        alpha_val = int(brightness) * 255
        color_with_alpha = (scaled_color[0], scaled_color[1], scaled_color[2], alpha_val)
        # cv2.circle(self.img, (int(x), int(y)), int(r), color_with_alpha, -1)
        cv2.fillConvexPoly(self.img, vertices, color_with_alpha)

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
        # instead of counting the digits below,
        # we could test if base_vector^2 = +1 or -1
        for digit in octal_str:
            if digit in '124':
                count += 1

        # Determine the order of the floretion
        order = len(octal_str)

        # Check the orientation based on the count and the order
        if count % 2 == 0:
            orientation = 'up' if order % 2 == 0 else 'down'
        else:
            orientation = 'down' if order % 2 == 0 else 'up'

        return orientation

    def plot_floretion(self):
        """
        Plots the floretion using either dots or triangles based on the plot type.
        """
        for base_vector, coeff in zip(self.base_vectors, self.floretion.coeff_vec_all):
            x, y, final_distance, color = self.place_dots(str(base_vector))
            brightness = abs(coeff)
            if brightness < .1:
                brightness = .1

            orientation = self.calculate_orientation(base_vector)

            if self.plot_type == 'dot' or self.plot_type == 'dots':
                self.draw_dot(x, y, brightness, color)
            elif self.plot_type == 'triangle' or self.plot_type == 'triangles':
                self.draw_triangle(x, y, final_distance, orientation, brightness, color)
                # Add the octal representation on top of the triangle
                cv2.putText(self.img, str(base_vector), (int(x)-10, int(y)+0), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 255, 255), 1)
            else:
                print(f"Unknown plottype {self.plot_type}")



    # ax.text(x, y, label, fontsize=12)

    def place_dots(self, base_vector):
        """
        Places dots on the canvas based on the given base vector and returns
        the coordinates, final distance, and color for further processing.

        Args:
            base_vector (str): The base vector in octal representation.

        Returns:
            tuple: x, y coordinates, final distance, and BGR color.
        """
        x, y = self.height // 2, self.width // 2
        y_offset = int(self.height * 0.1)  # Move down by 10% of the image height
        y += y_offset

        distance = self.height // self.distance_scale_fac
        sign_distance = -1

        green_value, red_value, blue_value = 0, 0, 0

        for digit in base_vector:
            # digit 4 should be at 330 and digit 1 at 210, i.e. 1 and 4 should be reversed,
            # but writing the other away around here prevents us from having to call flip
            if digit == '4':
                angle = 210
                x += np.cos(np.radians(angle)) * distance * sign_distance
                y += np.sin(np.radians(angle)) * distance * sign_distance
            elif digit == '2':
                angle = 90
                x += np.cos(np.radians(angle)) * distance * sign_distance
                y += np.sin(np.radians(angle)) * distance * sign_distance
            elif digit == '1':
                angle = 330
                x += np.cos(np.radians(angle)) * distance * sign_distance
                y += np.sin(np.radians(angle)) * distance * sign_distance
            elif digit == '7':
                sign_distance *= -1
            else:
                print(f"Invalid digit {digit}")

            # Calculate the distance to the bottom left corner (i...i)
            green_distance = np.sqrt((x - 0) ** 2 + (y - 0) ** 2)

            # Calculate the distance to the bottom right corner (j...j)
            red_distance = np.sqrt((x - width) ** 2 + (y - 0) ** 2)

            # Calculate the distance to the top corner (k...k)
            blue_distance = np.sqrt((x - width // 2) ** 2 + (y - height) ** 2)

            # Normalize distances
            max_distance = np.sqrt(height ** 2 + width ** 2)
            green_value = int((green_distance / max_distance) * 255)
            red_value = int((red_distance / max_distance) * 255)
            blue_value = int((blue_distance / max_distance) * 255)

            distance /= 2  # Halve the distance for the next iteration

        # return 1.9*distance instead of 2*distance to allow for a little space between the triangles
        return x, y, 1.8 * distance, (red_value, green_value, blue_value)


height, width = 1024, 1024

# Initialize VideoWriter
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') # FourCC code for MP4
# out = cv2.VideoWriter('sierpinski_animation.mp4', fourcc, 30, (width, height))


# Initial setup
height, width = 2048, 2048
floretion_instance = Floretion([0], [11], format_type="oct")
all_base_vecs = floretion_instance.grid_flo_loaded_data['oct']
num_coeffs = len(all_base_vecs)

# Generating mean and variance for Gauss functions. Let's adjust this part.
mean_var_list = [(mean, 10) for mean in np.linspace(100, 900, num=num_coeffs)]
img = np.zeros((height, width, 3), np.uint8)
# Main loop
for t in range(1, 1):
    coeffs_t = generate_coeffs(t, num_coeffs, mean_var_list)
    floretion_instance.coeff_vec_all = coeffs_t  # Floretion(coeffs_t, all_base_vecs, format_type="oct")
    # print(f"coeffs_t: {coeffs_t}")
    img_t = np.zeros((height, width, 4), np.uint8)
    sierpinski_instance = SierpinskiFlo(floretion_instance, img_t, plot_type='triangle')
    sierpinski_instance.plot_floretion()
    img_t = cv2.flip(img_t, 1)

    filedir = f"./data/anim_order_{floretion_instance.flo_order}/"
    filename = filedir + f"flo_img_{t}.png"

    if not os.path.exists(filedir):
        os.makedirs(filedir)

    if img_t.any():  # Check if the frame has any non-zero values
        print(f"Writing frame for t = {t}")
        cv2.imwrite(filename, img_t)
    else:
        print(f"Frame for t = {t} is all zeros.")

    # out.write(img_t)

# Release the VideoWriter
# out.release()

# Optional: Release any additional OpenCV objects
# cv2.destroyAllWindows()

# floretion_instance = Floretion.from_string(".2ii+ij+ik+ie+ei+ej+ek+ee+ji+jj+jk+je+ki+kj+kk+ke")


# floretion_instance = Floretion.from_string(".25iii+.25jjj+.25kkk+.25eei+.25ijk-eje+.2jij-eke")
# floretion_instance = floretion_instance*floretion_instance
# floretion_instance = floretion_instance*Floretion.from_string("eji-eke") + Floretion.from_string("jjj")
# floretion_instance = Floretion.from_string("iiii+jjjj+kkkk+eeee+eeej")
#floretion_instance = Floretion.from_string("iii + iie + iee + iei + eei + eee + eie + eii")
#floretion_instance = Floretion.from_string("jjj + jje + jee + jej + eej + eee + eje + ejj")
fac_x = 1
fac_y = 1
floretion_instance_x = Floretion.from_string(f"{fac_x}jjek + {fac_x}iiek + {fac_x}ekek + {fac_x}kkkk + {fac_x}kkke + {fac_x}kkee  + {fac_x}keke")
floretion_instance_y = Floretion.from_string(f"{fac_y}ekee+ {fac_y}eiee+ {fac_y}eiej +{fac_y}ejee+ {fac_y}ejei")
floretion_instance = floretion_instance_x*floretion_instance_y #+ floretion_instance_y*floretion_instance_x

print(f"x*y={floretion_instance.as_floretion_notation()}")
height, width = 1024, 1024
img = np.zeros((height, width, 3), np.uint8)

# Create SierpinskiFlo instance
sierpinski_instance = SierpinskiFlo(floretion_instance, img, plot_type='triangle')
sierpinski_instance.plot_floretion()
filedir = f"./data/base_folder/order_{floretion_instance.flo_order}"
#filename = filedir + f"x.k_axis_symm.order_{floretion_instance.flo_order}.png"

filename = filedir + f"x_times_y.k_axis_symm.order_{floretion_instance.flo_order}.png"


if not os.path.exists(filedir):
    os.makedirs(filedir)

cv2.imwrite(filename, img)

#img = cv2.flip(img, 1)
cv2.imshow('SierpinskiFlo', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img_x = np.zeros((height, width, 3), np.uint8)
img_x = np.zeros((height, width, 4), np.uint8)
img_x[..., 3] = 0  # 4th channel is alpha (0=fully transparent)

img_y = np.zeros((height, width, 3), np.uint8)
img_z = np.zeros((height, width, 3), np.uint8)

# flo_x = Floretion.from_string("iii + iie + iee + iei + eei + eee + eie + eii")
flo_x = Floretion.from_string("iii + .5ekk + jjj-jik")
sierpinski_instance = SierpinskiFlo(flo_x, img_x, plot_type='triangle')
sierpinski_instance.plot_floretion()

# flo_y= Floretion.from_string("jjj + jje + jee + jej + eej + eee + eje + ejj")
flo_y = Floretion.from_string("eje+eek+eie+iei+eke+jej")
sierpinski_instance = SierpinskiFlo(flo_y, img_y, plot_type='triangle')
sierpinski_instance.plot_floretion()

# flo_k = Floretion.from_string("kkk + kke + kee + kek + eek + eee + eke + ekk")
flo_z = flo_x * flo_y
sierpinski_instance = SierpinskiFlo(flo_z, img_z, plot_type='dot')
sierpinski_instance.plot_floretion()

# Display the first image
img_x = cv2.flip(img_x, 1)
# cv2.imshow('SierpinskiFlo_x', img_x)
# cv2.moveWindow('SierpinskiFlo_x', 0, 0)
cv2.imwrite('img_x.png', img_x)

# Display the second image
img_y = cv2.flip(img_y, 1)
# cv2.imshow('SierpinskiFlo_y', img_y)
# cv2.moveWindow('SierpinskiFlo_y', img_x.shape[1], 0)  # Move this window to the right of the first one

# Display the third image
img_z = cv2.flip(img_z, 1)
# cv2.imshow('SierpinskiFlo_x_times_y', img_z)
# cv2.moveWindow('SierpinskiFlo_x_times_y', 2 * img_x.shape[1], 0)  # Move this window to the right of the second one

cv2.waitKey(0)
cv2.destroyAllWindows()
