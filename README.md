# Floretions: A Playground for Algebra and Geometry

## Introduction

This repository serves as a playground for exploring the algebraic structure of Floretions—a unique number system with roots in base-8 digits 1,2,4,7. A formal definition is given in [OEIS](https://oeis.org/search?q=a308496).  "Linking Quaternions and the Sierpinski Gasket via Floretions: Equilateral Triangle Tiling and Multiplication Invariance." is one area of current research. 

## Content Overview

- `floretion.py`: Implements the `Floretion` class to perform fundamental operations like creation and multiplication.
- `SierpinskiFlo.py`: An example application that leverages Floretions to generate and display Sierpinski Gasket fractals.
- `data/`: Directory containing CSV files for various uses, including storing calculated data and parameters for fractal generation.

## Classes

### Floretion

#### Attributes

- `base_vectors`: List of Floretion base vectors.
- `coefficients`: List of coefficients for the Floretion object.

#### Methods

- `__init__(self, base_vectors, coefficients)`: Initializes a new `Floretion` object.
- `__mul__(self, other)`: Multiplies two `Floretion` objects.
- `__add__(self, other)`: Adds two `Floretion` objects.

### SierpinskiFlo

#### Methods

- `generate_fractal()`: Generates a Sierpinski Gasket using Floretions.
- `display_fractal()`: Displays the generated Sierpinski Gasket.

## Example Usage

```python
from floretion import Floretion

f1 = Floretion([...], [...])
f2 = Floretion([...], [...])
result = f1 * f2
print(result)

from SierpinskiFlo import SierpinskiFlo

s = SierpinskiFlo()
s.generate_fractal()
s.display_fractal()
