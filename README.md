# Floretion Base Vector

## Introduction

This repository contains Python implementations of Floretion base vectors, a formal definition can be found at [OEIS](https://oeis.org/search?q=a308496). Floretions are numbers written in base 8 with digits 1,2,4,7. They form a mathematical structure that can be manipulated and studied much like more common algebraic structures.

## How it Works

### The Base Vector Class

The `floretion_base_vector` class defines a Floretion base vector and provides methods for its manipulation.

#### Attributes

- `value`: Integer
  - The decimal representation of the Floretion base vector.
- `order`: Integer
  - The order of the Floretion base vector, determined by its length in octal representation.

#### Methods

- `__init__(self, x)`: Initializes a new `floretion_base_vector` object.
- `determine_order(self)`: Determines the order of the Floretion base vector.
- `as_octal(self)`: Returns the octal representation of the Floretion base vector.
- `as_decimal(self)`: Returns the decimal representation of the Floretion base vector.
- `as_binary(self)`: Returns the binary representation of the Floretion base vector.
- `as_floretion_notation(self)`: Returns the floretion notation, a human-readable form.
- `get_order(self)`: Returns the order of the Floretion base vector.
- `__mul__(self, other)`: Multiplies two `floretion_base_vector` objects.

### Static Methods

- `mult_flo(a_base_val, b_base_val, flo_order)`: Computes the product of two Floretion base vectors given their base values and order.

## Example Usage

```python
flo1 = floretion_base_vector(-84095)
print("-84095 in floretion notation:", flo1.as_floretion_notation())

flo2 = floretion_base_vector(145)
flo3 = floretion_base_vector(143)
print("flo2*flo3 in floretion notation:", (flo2 * flo3).as_floretion_notation())
