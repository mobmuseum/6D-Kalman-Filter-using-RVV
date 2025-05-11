import sys
import random

def generate_matrix(size):
    """Generates a square matrix of given size with random floats between -1000 and 1000, rounded to 0.25 increments."""
    matrix = []
    for _ in range(size):
        row = [round(random.uniform(-1000, 1000) * 4) / 4 for _ in range(size)]
        matrix.append(row)
#     matrix = [
#     [-408.25, -504.5, -639.75, -934.25, 345.0],
#     [590.0, 583.5, 598.25, -628.0, -798.5],
#     [-557.25, 564.25, 201.0, -442.75, 970.0],
#     [198.75, 614.0, -253.25, -111.25, -304.0],
#     [-48.75, 333.5, 785.5, 347.5, 393.5]
# ]
    return matrix

def read_and_modify_file(filename, matrix, size, type):
    """Reads the file, replaces the matrix data, and writes a modified version."""
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Find the markers
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "## DATA DEFINE START":
            start_idx = i
        elif line.strip() == "## DATA DEFINE END":
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        raise ValueError("Start or end marker not found in the file.")

    # Create the matrix data
    matrix_data = [f".equ MatrixSize, {size}\n", "matrix:\n"]
    for row in matrix:
        row_data = ", ".join(map(str, row))
        matrix_data.append(f"    .float {row_data}\n")

    # Replace the data between the markers
    modified_lines = lines[:start_idx + 1] + matrix_data + lines[end_idx:]

    # Write the modified file
    if type == "V":
        savefilename =  "assembly/VectorizedModified.s"
    else:
        savefilename = "assembly/NonVectorizedModified.s"
    
    with open(savefilename, 'w') as file:
        file.writelines(modified_lines)

def write_to_file(size, type):
    try:
        if size <= 0:
            raise ValueError("Size must be a positive integer.")

        if type == "V":
            filename =  "assembly/Vectorized.s"
        else:
            filename = "assembly/NonVectorized.s"
        matrix = generate_matrix(size)
        read_and_modify_file(filename, matrix, size, type)
        print(f"File successfully modified and saved as 'modified'.")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    except FileNotFoundError:
        print("Error: The specified file was not found.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <size> <type>")
        sys.exit(1)
    size = int(sys.argv[1])
    type = sys.argv[2]

    write_to_file(size, type)

