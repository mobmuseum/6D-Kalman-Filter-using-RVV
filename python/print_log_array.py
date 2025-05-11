import math
import struct
import sys

def find_and_save_lines(filename):
    all_results = []
    result = []
    size_line = ""
    is_saving = False

    with open(filename, 'r') as file:
        prev_line = None  # Store the previous line for pattern matching

        for line in file:
            line_strip = line.strip()
            columns = line_strip.split()

            if prev_line:
                prev_columns = prev_line.strip().split()

                if len(prev_columns) > 6 and len(columns) > 6:
                    if prev_columns[6] == "00000123" and columns[6] == "00000456":
                        if is_saving:  # If already saving, store result and reset
                            all_results.append(result)
                            result = []
                            is_saving = False
                        else:
                            is_saving = True

            if is_saving:
                if "c.mv     a1" in line_strip:
                    size_line = line_strip
                result.append(line_strip)

            prev_line = line  # Move to the next line

    if result:
        all_results.append(result)

    return all_results, size_line


def filter_lines_by_flw(lines):
    return [line for line in lines if "flw" in line]


def filter_lines_by_vle(lines):
    return [line for line in lines if "vle32.v" in line]


def extract_7th_column(lines):
    return [line.strip().split()[6] for line in lines if len(line.strip().split()) > 6]



def hex_to_float(hex_array, isVectorized=False, size=None):
    result = []
    
    for hex_str in hex_array:
        if isVectorized:
            # Split into 8-character (32-bit) chunks
            chunks = [hex_str[i:i+8] for i in range(0, len(hex_str), 8)]
            # Reverse the order (as per RISC-V vector register format)
            chunks.reverse()
        else:
            chunks = [hex_str]
        
        for chunk in chunks:
            if len(chunk) != 8:
                continue  # Skip invalid chunks
            try:
                hex_value = int(chunk, 16)
                float_value = struct.unpack('!f', struct.pack('!I', hex_value))[0]
                result.append(float_value)
            except ValueError:
                pass  # Ignore invalid hex values
    
    # Trim to the specified size if needed
    if size is not None:
        result = result[:size]
    
    return result


def print_matrixes_from_lines(lines, size):
    isVector = any("vle" in line for array in lines for line in array)

    arr = [hex_to_float(extract_7th_column(filter_lines_by_flw(matrix))) for matrix in lines]
    if (isVector):
        arr = [hex_to_float(extract_7th_column(filter_lines_by_vle(matrix)), isVectorized=isVector, size=size**2) for matrix in lines]
    

    print("Matrixes are:\n")
    for array in arr:
        for i in range(0, len(array), size):
            print(" ".join(f"{x:12.6f}" for x in array[i:i + size]))
        print("")


if __name__ == "__main__":
    size = int(sys.argv[1])
    type = sys.argv[2]
    fileName = "veer/tempFiles/logNV.txt" if type == "NV" else "veer/tempFiles/logV.txt"

    a, size_line = find_and_save_lines(fileName)
    if (not a) or (not size_line):
        print("Not found")
        exit()
    size = int(extract_7th_column([size_line])[0], 16)
    print("matrix size is ", size)
    print_matrixes_from_lines(a, size)
