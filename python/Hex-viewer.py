import struct

filename = "output.hex"
floats_to_read = 6

try:
    with open(filename, "rb") as f:
        print(f"Contents of {filename}:")
        for i in range(floats_to_read):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Reached end of file prematurely.")
                break
            value = struct.unpack('>f', four_bytes)[0]
            hex_representation = four_bytes.hex().upper()
            print(f"Float {i+1}: Hex = 0x{hex_representation}, Decimal = {value}")
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")