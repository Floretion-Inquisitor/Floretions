# Floretions: A Playground for Algebra and Geometry

## Introduction

This repository serves as a playground for exploring the algebraic structure of Floretions—a unique number system with roots in base-8 digits 1,2,4,7. [Floretion Calculator](https://www.floretions.com).  

## Content Overview

- `floretion.py`: Implements the `Floretion` class to perform fundamental operations like creation and multiplication.
- `SierpinskiFlo.py`: Pass this class an object of type `Floretion` to be represented as a triangle tiling.
- `data/`: Directory containing CSV files to directly import base vectors of order 9 and less (as opposed to calculating these "on the fly", which could take a lot of time). 

## Classes

### Class: Floretion
Represents a Floretion, a type of hypercomplex number that extends the concept of quaternions.

#### Attributes
- `base_to_nonzero_coeff`: Maps base vectors (in decimal) to their non-zero coefficients.
- `format_type`: String representing the format ('dec' or 'oct') of base vectors.
- `max_order`: Maximum order for the floretions.
- `flo_order`: The floretion order.
- `grid_flo_loaded_data`: Pandas DataFrame holding loaded data.
- `base_vec_dec_all`: NumPy array of all base vectors in decimal.
- `coeff_vec_all`: NumPy array of coefficients aligned with `base_vec_dec_all`.
- `base_to_grid_index`: Maps base vectors to their grid indices.

#### Methods
- `__init__`: Initializes a Floretion instance.
- `from_preloaded_data`, `from_cartesian_coords`: Methods for creating Floretions from different data types.
- `find_flo_order`: Calculates the floretion order.
- Arithmetic operations: `__pow__`, `__add__`, `__sub__`, `__eq__`, `__mul__`, `__rmul__`.
- `mult_flo_base_absolute_value`, `mult_flo_sign_only`: Methods for multiplying floretion base vectors.
- `compute_possible_vecs`: Computes possible vectors.
- `as_floretion_notation`: Converts to floretion notation.
- `flo_oct_to_grid`, `grid_to_flo_oct`: Conversion between floretion octal representation and grid format.
- `normalize_coeffs`: Normalizes coefficients.
- `sum_of_squares`, `abs`: Calculates sum of squares and absolute value.
- `from_string`: Creates a Floretion from a string representation.
- `display_as_grid`: Displays the Floretion as a grid.
- `grid_to_coordinates`: Converts grid format to coordinates.
- `find_center`: Finds the center of a Floretion.


## Example Usage

    flo_x = Floretion.from_string(".5iii +.5jjj + .5kkk")
    flo_y = Floretion.from_string("iij + iik + jji +jjk + kki + kkj")

    flo_z = flo_x * flo_y - flo_y * flo_x
    print(f"flo_z {flo_z.as_floretion_notation()}")
