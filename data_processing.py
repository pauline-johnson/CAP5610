# This file processes a CSV file and returns a list of tuples with its data.

# Returns a list of tuples representing each line in the data given a file name.
def load_data(filename):

    # Opens and read the file as text. Saves each line in a list.
    file_handler = open(filename, "rt")
    lines = file_handler.readlines()
    file_handler.close()

    # Creates a list of tuples from the list of lines.
    dataset = []
    labels = []
    for line in lines:
        instance, label = line_to_tuple(line)
        dataset.append(instance)
        labels.append(label)

    return dataset, labels

# Returns a tup
def line_to_tuple(line):

    # Strip line of whitespace and newlines.
    clean_line = line.strip()

    # Split by comma into a list of string attributes.
    line_list = clean_line.split(",")

    # Convert the string attributes to numerical ones.
    label = string_to_num(line_list)

    # Cast list to tuple.
    line_tuple = tuple(line_list)

    return line_tuple, label

# Converts the numerical items in a given list to floats.
# Dispose of any non-numerical attributes for now.
def string_to_num(line_list):
    for i in range(len(line_list)):
        if is_number(line_list[i]):
            line_list[i] = float(line_list[i])
        else:
            return line_list.pop(i)


def is_number(num):
    if len(num) == 0:
        return False
    if len(num) > 1 and num[0] == "-":
        num = num[1:]
    for c in num:
        if c not in "0123456789.":
            return False
    return True

