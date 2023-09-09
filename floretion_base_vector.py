def count_bits(n):
    return bin(n).count('1')


def sgn(x):
    return -1 if x < 0 else 1


class floretion_base_vector:
    """
    A class to represent the base vector of a floretion.

    Attributes
    ----------
    value : int
        The octal value representing the floretion base vector.
    order : int
        The order of the floretion, determined by the length of the octal representation.

    Methods
    -------
    determine_order():
        Determines the order of the floretion based on its octal representation.
    as_octal():
        Returns the octal representation of the floretion base vector as a string.
    as_decimal():
        Returns the decimal representation of the floretion base vector as a string.
    as_binary():
        Returns the binary representation of the floretion base vector as a string.
    as_floretion_notation():
        Returns the floretion notation (e.g., 'ijke') of the floretion base vector.
    get_order():
        Returns the order of the floretion.

    Example
    -------
    >>> f = floretion_base_vector(84095)
    >>> f.as_floretion_notation()
    'jkkiee'
    """

    def __init__(self, x):
        """
        Initializes a floretion_base_vector object.

        Parameters:
        x (int): The base vector value.

        Raises:
        ValueError: If x is not a valid floretion base vector.
        """
        if not all(digit in '1247' for digit in oct(abs(x))[2:]):
            raise ValueError(f"{x} is not a valid floretion base vector")
        self.value = x
        self.order = self.determine_order()

    def determine_order(self):
        """
                Determines the order of the floretion.

                Returns:
                int: The order of the floretion.
        """
        return len(oct(self.value)[2:])

    def as_octal(self):
        """
               Returns the floretion in octal format.

               Returns:
               str: The octal representation of the floretion.
        """
        octal_str = oct(abs(self.value))[2:]
        return "_".join(octal_str)


    def as_decimal(self):
        """
                Returns the floretion in decimal format.

                Returns:
                str: The decimal representation of the floretion.
        """
        return str(self.value)

    def as_binary(self):
        """
                Returns the floretion in binary format.

                Returns:
                str: The binary representation of the floretion.
        """
        binary_str = bin(self.value)[2:].zfill(self.order * 3)
        return "_".join(binary_str[i:i + 3] for i in range(0, len(binary_str), 3))

    def as_floretion_notation(self):
        """
                Returns the floretion in floretion notation.

                Returns:
                str: The floretion notation.
        """
        decimal_mapping = {1: 'i', 2: 'j', 4: 'k', 7: 'e'}
        octal_str = self.as_octal().replace("_", "")
        notation = ''.join([decimal_mapping[int(ch)] for ch in octal_str])
        if self.value < 0:
            notation = "-" + notation
        return notation


    def get_order(self):
        """
        Retrieves the order of the floretion.

        Returns:
        int: The order of the floretion.
        """
        return self.order


    def __mul__(self, other):
        """
        Multiplies two floretion_base_vectors.

        Parameters:
        other (floretion_base_vector): Another floretion_base_vector object.

        Returns:
        floretion_base_vector: A new floretion_base_vector that's the result of the multiplication.

        Raises:
        ValueError: If the orders of the two floretions don't match.
        """
        if self.order != other.order:
            raise ValueError("Floretions must be of the same order to multiply")

        result_value =  floretion_base_vector.mult_flo(self.value, other.value, self.order)

        # Handle the sign of the result if needed

        return floretion_base_vector(result_value)


    @staticmethod
    def mult_flo(a_base_val, b_base_val, flo_order):
        """
          Multiplies two floretion base vectors of the same order.

          Parameters
          ----------
          a_base_val : int
              The base vector value of the first floretion.
          b_base_val : int
              The base vector value of the second floretion.
          flo_order : int
              The order of the floretions being multiplied.

          Returns
          -------
          int
              The resulting floretion base vector after multiplication, with the signs considered.

          Example
          -------
          >>> floretion_base_vector.mult_flo(145, 143, 2)
          -33

          Notes
          -----
          The multiplication is performed based on bitwise logical operations and specific sign rules.
        """
        bitmask = int(2 ** (3 * flo_order) - 1)
        OCT666 = int('6' * flo_order, 8)
        OCT111 = int('1' * flo_order, 8)

        pre_sign = sgn(a_base_val) * sgn(b_base_val)
        a_base_val = abs(a_base_val)
        b_base_val = abs(b_base_val)

        # Shift every 3-bits of "a" one to the left
        a_cyc = ((a_base_val << 1) & OCT666) | ((a_base_val >> 2) & OCT111)

        cyc_sign = 1 if count_bits((a_cyc & b_base_val) & bitmask) & 0b1 else -1
        ord_sign = 1 if count_bits(bitmask) & 0b1 else -1

        return (pre_sign * cyc_sign * ord_sign) * (bitmask & (~(a_base_val ^ b_base_val)))

    @staticmethod
    def mult_flo_sign_only(a_base_val, b_base_val, flo_order):
        bitmask = int(2 ** (3 * flo_order) - 1)
        OCT666 = int('6' * flo_order, 8)
        OCT111 = int('1' * flo_order, 8)

        pre_sign = sgn(a_base_val) * sgn(b_base_val)
        a_base_val = abs(a_base_val)
        b_base_val = abs(b_base_val)

        # Shift every 3-bits of "a" one to the left
        a_cyc = ((a_base_val << 1) & OCT666) | ((a_base_val >> 2) & OCT111)

        cyc_sign = 1 if count_bits((a_cyc & b_base_val) & bitmask) & 0b1 else -1
        ord_sign = 1 if count_bits(bitmask) & 0b1 else -1

        return (pre_sign * cyc_sign * ord_sign)


if __name__ == "__main__":
    flo1 = floretion_base_vector(-84095)

    #print("84095 in octal:", flo1.as_octal())
    #print("84095 in binary:", flo1.as_binary())
    print("-84095 in floretion notation:", flo1.as_floretion_notation())
    #print("Order of flo1:", flo1.get_order())

    flo2 = floretion_base_vector(145)
    flo3 = floretion_base_vector(143)
    print("flo2:", flo2.as_floretion_notation())
    print("flo3:", flo3.as_floretion_notation())
    #print("Order of flo2:", flo2.get_order())
    flo4 = floretion_base_vector.__mul__(flo2, flo3)
    print("flo2*flo3 in floretion notation:", flo4.as_floretion_notation())

    print(floretion_base_vector.mult_flo_sign_only(1,1,1))