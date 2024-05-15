from flask import Flask, request, jsonify, send_file, render_template
from floretion import Floretion
from triangleize import Triangleize
import cv2
import numpy as np
from io import BytesIO
import re

app = Flask(__name__)

MAX_FLO_ORDER = 6
MAX_FLORETION_LENGTH = 100000

# Configuration variables
APP_CONFIG = {
    'background_color': '#ccffff',
    'font_size': '14px'
}

def sanitize_input(input_str):
    """Sanitize and validate input strings."""
    if len(input_str) > MAX_FLORETION_LENGTH:
        raise ValueError(f"Input exceeds maximum allowed length ({MAX_FLORETION_LENGTH} characters).")
    return input_str.strip()

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
        print(base_vec_oct_str)
        storage_type = "json"
        if command == "Cp":
            centers_data = Floretion.load_centers(floretion_base_vec.flo_order, decomposition_type="pos", storage_type=storage_type)
        elif command == "Cn":
            centers_data = Floretion.load_centers(floretion_base_vec.flo_order, decomposition_type="neg", storage_type=storage_type)
        elif command == "Cb":
            centers_data = Floretion.load_centers(floretion_base_vec.flo_order, decomposition_type="both", storage_type=storage_type)
        else:
            raise ValueError("Unknown command type. Use Cp, Cn, or Cb.")
        print(centers_data)
        coeff_array = np.ones(len(centers_data))
        return Floretion(coeff_array, np.array(centers_data[base_vec_oct_str]))
    else:
        # Check if input contains valid characters
        if not all(c in "0123456789ijke.+ -" for c in input_str):
            raise ValueError("Invalid character in floretion string.")
        return Floretion.from_string(input_str)

def summarize_floretion(flo_str):
    """Return a summary of a large Floretion string."""
    if len(flo_str) <= 100:
        return flo_str
    return f"{flo_str[:50]}...{flo_str[-50:]}"

def decimal_to_octal(decimal):
    return format(int(decimal), 'o')

def get_typical_floretions(order):
    zero_flo = Floretion.from_string(f'0{"e" * order}')
    unit_flo = Floretion.from_string(f'1{"e" * order}')
    new_coeffs_sierp = []
    new_coeffs_sierp_i = []
    new_coeffs_sierp_j = []
    new_coeffs_sierp_k = []

    axis_i = []
    axis_j = []
    axis_k = []

    for base in zero_flo.base_vec_dec_all:
        base_octal = decimal_to_octal(base)

        new_coeffs_sierp.append(0.0 if '7' in base_octal else 1.0)
        new_coeffs_sierp_i.append(0.0 if '1' in base_octal else 1.0)
        new_coeffs_sierp_j.append(0.0 if '2' in base_octal else 1.0)
        new_coeffs_sierp_k.append(0.0 if '4' in base_octal else 1.0)

        axis_i.append(0.0 if '2' in base_octal or '4' in base_octal else 1.0)
        axis_j.append(0.0 if '4' in base_octal or '1' in base_octal else 1.0)
        axis_k.append(0.0 if '1' in base_octal or '2' in base_octal else 1.0)

    new_coeffs_sierp = np.array(new_coeffs_sierp)
    new_coeffs_sierp_i = np.array(new_coeffs_sierp_i)
    new_coeffs_sierp_j = np.array(new_coeffs_sierp_j)
    new_coeffs_sierp_k = np.array(new_coeffs_sierp_k)

    coeffs_axis_i = np.array(axis_i)
    coeffs_axis_j = np.array(axis_j)
    coeffs_axis_k = np.array(axis_k)

    norm_fac = 1.  # (np.sqrt(1 / 3) ** order)
    sierp_flo = Floretion(norm_fac * new_coeffs_sierp, zero_flo.base_vec_dec_all, format_type="dec")
    sierp_flo_i = Floretion(norm_fac * new_coeffs_sierp_i, zero_flo.base_vec_dec_all, format_type="dec")
    sierp_flo_j = Floretion(norm_fac * new_coeffs_sierp_j, zero_flo.base_vec_dec_all, format_type="dec")
    sierp_flo_k = Floretion(norm_fac * new_coeffs_sierp_k, zero_flo.base_vec_dec_all, format_type="dec")

    axis_i = Floretion(coeffs_axis_i, zero_flo.base_vec_dec_all, format_type="dec")
    axis_j = Floretion(coeffs_axis_j, zero_flo.base_vec_dec_all, format_type="dec")
    axis_k = Floretion(coeffs_axis_k, zero_flo.base_vec_dec_all, format_type="dec")

    axis_ij = axis_i + axis_j - unit_flo
    axis_jk = axis_j + axis_k - unit_flo
    axis_ki = axis_k + axis_i - unit_flo
    axis_ijk = axis_i + axis_j + axis_k - 2*unit_flo

    typical_floretions = {
        "unit": unit_flo.as_floretion_notation(),
        "axis-I": axis_i.as_floretion_notation(),
        "axis-J": axis_j.as_floretion_notation(),
        "axis-K": axis_k.as_floretion_notation(),
        "axis-IJ": axis_ij.as_floretion_notation(),
        "axis-JK": axis_jk.as_floretion_notation(),
        "axis-KI": axis_ki.as_floretion_notation(),
        "axis-IJK": axis_ijk.as_floretion_notation(),
        "sierpinski-E": sierp_flo.as_floretion_notation(),
        "sierpinski-I": sierp_flo_i.as_floretion_notation(),
        "sierpinski-J": sierp_flo_j.as_floretion_notation(),
        "sierpinski-K": sierp_flo_k.as_floretion_notation()
    }
    return {
        name: {"summary": value, "full": value}
        for name, value in typical_floretions.items()
    }


@app.route('/')
def index():
    return render_template('index.html', config=APP_CONFIG)

@app.route('/calculate_floretion', methods=['POST'])
def calculate_floretion():
    try:
        data = request.json
        x_str = sanitize_input(data['x'])
        y_str = sanitize_input(data['y'])
        order = int(data['order'])
        x_slider = float(data.get('x_slider', 1))
        y_slider = float(data.get('y_slider', 1))

        print(f"Received x_slider: {x_slider}, y_slider: {y_slider}")  # Debugging

        # Custom handling for Cp, Cn, and Cb commands
        x_floretion = parse_special_commands(x_str, order)
        y_floretion = parse_special_commands(y_str, order)

        if x_floretion.flo_order != y_floretion.flo_order:
            raise ValueError(f"Incompatible orders: {x_floretion.flo_order} vs {y_floretion.flo_order}")

        result_fac = x_slider * y_slider
        result_floretion = result_fac * (x_floretion * y_floretion)

        result_str = result_floretion.as_floretion_notation()

        return jsonify({'result': summarize_floretion(result_str), 'full_result': result_str, 'order': order})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/floretion_image_input', methods=['POST'])
def floretion_image():
    try:
        data = request.json
        floretion_str = sanitize_input(data['floretion'])
        order = int(data.get('order', 1))  # Provide a default value if 'order' is missing
        x_slider = float(data.get('x-slider', 1))
        y_slider = float(data.get('y-slider', 1))

        floretion = parse_special_commands(floretion_str, order)


        img = np.zeros((1600, 1600, 3), dtype=np.uint8)  # Adjust size here

        floA = Triangleize(floretion, img, plot_type='triangle')
        floA.plot_floretion()

        is_success, img_buffer = cv2.imencode('.png', img)
        img_io = BytesIO(img_buffer.tobytes())
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        print(f"Error generating preview image: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/floretion_image_result', methods=['POST'])
def floretion_image_result():
    try:
        data = request.json
        floretion_str = sanitize_input(data['floretion'])
        order = int(data.get('order', 1))  # Provide a default value if 'order' is missing

        floretion = parse_special_commands(floretion_str, order)

        img = np.zeros((800, 800, 3), dtype=np.uint8)  # Adjust size here

        floA = Triangleize(floretion, img, plot_type='triangle')
        floA.plot_floretion()

        is_success, img_buffer = cv2.imencode('.png', img)
        img_io = BytesIO(img_buffer.tobytes())
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        print(f"Error generating result image: {e}")
        return jsonify({'error': str(e)}), 400



@app.route('/validate_floretion', methods=['POST'])
def validate_floretion():
    """Check if the given floretion string is valid without loading an image."""
    try:
        data = request.json
        floretion_str = sanitize_input(data['floretion'])
        order = int(data['order'])

        parse_special_commands(floretion_str, order)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/typical_floretions', methods=['GET'])
def typical_floretions():
    try:
        order = int(request.args.get('order', 1))
        floretion_map = get_typical_floretions(order)
        return jsonify({
            'unit': floretion_map["unit"]["summary"],
            'x': [
                {"name": "sierpinski-E", "summary": floretion_map["sierpinski-E"]["summary"], "full": floretion_map["sierpinski-E"]["full"]},
                {"name": "sierpinski-I", "summary": floretion_map["sierpinski-I"]["summary"], "full": floretion_map["sierpinski-I"]["full"]},
                {"name": "sierpinski-J", "summary": floretion_map["sierpinski-J"]["summary"], "full": floretion_map["sierpinski-J"]["full"]},
                {"name": "sierpinski-K", "summary": floretion_map["sierpinski-K"]["summary"], "full": floretion_map["sierpinski-K"]["full"]},
                {"name": "axis-I", "summary": floretion_map["axis-I"]["summary"], "full": floretion_map["axis-I"]["full"]},
                {"name": "axis-J", "summary": floretion_map["axis-J"]["summary"], "full": floretion_map["axis-J"]["full"]},
                {"name": "axis-K", "summary": floretion_map["axis-K"]["summary"], "full": floretion_map["axis-K"]["full"]},
                {"name": "axis-IJ", "summary": floretion_map["axis-IJ"]["summary"], "full": floretion_map["axis-IJ"]["full"]},
                {"name": "axis-JK", "summary": floretion_map["axis-JK"]["summary"], "full": floretion_map["axis-JK"]["full"]},
                {"name": "axis-KI", "summary": floretion_map["axis-KI"]["summary"], "full": floretion_map["axis-KI"]["full"]},
                {"name": "axis-IJK", "summary": floretion_map["axis-IJK"]["summary"], "full": floretion_map["axis-IJK"]["full"]}
            ],
            'y': [
                {"name": "sierpinski-E", "summary": floretion_map["sierpinski-E"]["summary"], "full": floretion_map["sierpinski-E"]["full"]},
                {"name": "sierpinski-I", "summary": floretion_map["sierpinski-I"]["summary"], "full": floretion_map["sierpinski-I"]["full"]},
                {"name": "sierpinski-J", "summary": floretion_map["sierpinski-J"]["summary"], "full": floretion_map["sierpinski-J"]["full"]},
                {"name": "sierpinski-K", "summary": floretion_map["sierpinski-K"]["summary"], "full": floretion_map["sierpinski-K"]["full"]},
                {"name": "axis-I", "summary": floretion_map["axis-I"]["summary"], "full": floretion_map["axis-I"]["full"]},
                {"name": "axis-J", "summary": floretion_map["axis-J"]["summary"], "full": floretion_map["axis-J"]["full"]},
                {"name": "axis-K", "summary": floretion_map["axis-K"]["summary"], "full": floretion_map["axis-K"]["full"]},
                {"name": "axis-IJ", "summary": floretion_map["axis-IJ"]["summary"], "full": floretion_map["axis-IJ"]["full"]},
                {"name": "axis-JK", "summary": floretion_map["axis-JK"]["summary"], "full": floretion_map["axis-JK"]["full"]},
                {"name": "axis-KI", "summary": floretion_map["axis-KI"]["summary"], "full": floretion_map["axis-KI"]["full"]},
                {"name": "axis-IJK", "summary": floretion_map["axis-IJK"]["summary"], "full": floretion_map["axis-IJK"]["full"]}
            ]
        })
    except Exception as e:
        print(f"Error fetching typical floretions: {e}")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
