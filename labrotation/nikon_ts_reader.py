from typing import TextIO, List, Tuple

# TODO: write tests for correct_element_m_s_ms!!!

STANDARD_COLUMN_NAMES = ["Time [s]", "SW Time [s]", "NIDAQ Time [s]", "Index"]
N_STANDARD_COLUMNS = len(STANDARD_COLUMN_NAMES)  # there are 4 standard columns


def read_contents(nik_file: TextIO) -> Tuple[List[str], List[List[float]]]:
    """
    Given an opened text file (result of open() ), read out title and data.
    Example:
    with open(filename, "r", encoding="utf-8") as f:
        res = read_contents(f)
    """
    title = nik_file.readline().strip()  # read first line: column names
    if len(title) > len(STANDARD_COLUMN_NAMES):
        print("Too many column names detected. Correcting to standard 4-column format")
        title = STANDARD_COLUMN_NAMES.copy()
    rows = nik_file.readlines()  # read the rest: numerical data
    # remove whitespaces, trailing \n
    rows = list(map(lambda row: row.strip().split("\t"), rows))
    rows = decimal_comma_to_dot(rows)
    rows = correct_listlist_m_s_ms(rows)
    rows = remove_stim_frames(rows)
    rows = convert_to_float(rows)
    if is_consistent(rows):
        print("Consistency test succeeded.")
    else:
        raise Exception("Consistency test failed!")
    return (title, rows)


def remove_stim_frames(rows: List[List[str]]) -> List[List[str]]:
    """
    Remove the frames that are - supposedly - stimulation frames, based on the length of these frames.
    Returns the 2d list without these frames.
    """
    n_removed_rows = 0
    i = 0
    while i < len(rows):
        if len(rows[i]) != N_STANDARD_COLUMNS:
            print(f"Removing column:\n {', '.join(rows[i])}")
            rows.pop(i)
            n_removed_rows += 1
        else:
            i += 1
    print(f"Removed {n_removed_rows} stimulation frames.")
    return rows


def test_length(rows: List[List[str]]) -> bool:
    for row in rows:
        if len(row) > 4:
            print(f"Fail: {row}")


def is_consistent(rows: List[List[str]]) -> bool:
    passed_test = True
    # First row too many entries
    if len(rows[0]) != N_STANDARD_COLUMNS:
        print("First row does not have the standard entries!")
        passed_test = False
    for i in range(1, len(rows)):
        # Row has too many entries
        if len(rows[i]) != N_STANDARD_COLUMNS:
            print(
                f"Row {i} does not have standard entries:\n  {', '.join(rows[i])}")
            passed_test = False
        # Time (s) not monotonically increasing
        if rows[i][0] <= rows[i-1][0]:
            print(
                f"Time (s) column {i-1} to {i} (0-indexing) not monotonically increasing!\n  {', '.join(rows[i-1])}\n vs\n  {', '.join(rows[i])}")
        # SW Time (s) not monotonically increasing
        if rows[i][1] <= rows[i-1][1]:
            print(
                f"SW Time (s) column {i-1} to {i} (0-indexing) not monotonically increasing!\n  {', '.join(rows[i-1])}\n vs\n  {', '.join(rows[i])}")
            passed_test = False
        # NIDAQ Time (s) not monotonically increasing
        if rows[i][2] <= rows[i-1][2]:
            print(
                f"NIDAQ Time (s) column {i-1} to {i} (0-indexing) not monotonically increasing!\n  {', '.join(rows[i-1])}\n vs\n  {', '.join(rows[i])}")
            passed_test = False
        # Frame index not monotonically increasing
        if rows[i][3] <= rows[i-1][3]:
            print(
                f"Frame index column {i-1} to {i} (0-indexing) not monotonically increasing!\n  {', '.join(rows[i-1])}\n vs\n  {', '.join(rows[i])}")
            passed_test = False
    return passed_test


def convert_to_float(rows: List[List[str]]) -> List[List[float]]:
    return list(map(lambda row:  [float(element) for element in row], rows))


def decimal_comma_to_dot(rows_list: List[str]) -> List[str]:
    """
    Convert the decimal commas to dots in a list of rows which
    are lists of string entries themselves.
    Example input:
    [
    ["1,523", "1.2", "1"],
    ["1.5", "2", "9,99"],
    ]
    Example output:
    [
    ["1.523", "1.2", "1"],
    ["1.5", "2", "9.99"]
    ]
    """
    return list(map(lambda row_strings: [row_strings[i].replace(",", ".") for i in range(len(row_strings))], rows_list))


def correct_element_m_s_ms(element: str) -> str:
    """
    Convert a string of format m:s.ms into s.ms (with minutes converted into
    seconds)
    Example input:
    "01:02.345"
    Example output:
    "62.345"
    """
    m_sms = element.split(":")  # should contain only one ":" -> [m, s.ms]
    s_ms = m_sms[1].split(".")  # split s.ms into s and ms
    m = m_sms[0]
    s = s_ms[0]
    ms = s_ms[1]
    # format 0.12345 to 4 decimal places, and drop 0 in beginning (.1234); append it to integer part.
    return str(int(m)*60 + int(s)) + "{:.4f}".format(float("0." + ms))[1:]


def correct_row_m_s_ms(row: List[str]) -> List[str]:
    """
    Given a row of string values representing numbers, find - if there is any -
    an expression in the form of m:s.ms, and convert it to s.ms
    Example input:
    ["0", "1,4", "1.5", "05:12.1234"]
    Example output:
    ["0", "1,4", "1.5", "312.1234"]
    """
    row_corrected = row.copy()  # TODO: does in-place replacement work? (i.e. do not return anything, but change row; does it change the original row passed as an argument? Or, work with row and remove this line, and return row in the end?)
    for i in range(len(row)):
        if ":" in row_corrected[i]:  # found a weird entry of format m:s.ms
            row_corrected[i] = correct_element_m_s_ms(row[i])
    return row_corrected


def correct_listlist_m_s_ms(rows_list: List[List[str]]) -> List[List[str]]:
    """
    Find entries of the form m:s.ms in a list of list of strings and convert
    them to s.ms
    Example input:
    [
    ["01:05.1294", "1.5", "1", "1,3"],
    ]
    Example output:
    [
    ["65.1294", "1.5", "1", "1,3"],
    ]
    """
    return list(map(lambda row_strings: correct_row_m_s_ms(row_strings), rows_list))


def standardize_stamp_file(fname: str, export_fname: str, export_encoding: str = "utf-8") -> None:
    """
    Open a Nikon metadata file defined by fname (full path to a time stamps file, mostly ending with _nik.txt)
    and bring it to a standardized form, then save it in a different file defined by export_fname (full path to output file) with export_encoding.
    Standardized form is how the Nikon Elements exports normal imaging data (this works well with Matlab). When using
    stimulating lasers, for example, new columns appear, and weird changes (decimal comma instead of decimal point, but
    only in some places, weird time format, new columns)
    Steps to do:
        1.  Check number of columns. Leave the file unchanged if the column names are:
                Time [s], SW Time [s], NIDAQ Time [s], Index
            If columns are:
                Time [m:s.ms], SW Time [s], NIDAQ Time [s], Index, Events, Events Type
            Then start correction.
        2.  Load data as Header and Data.
        3.  Remove rows of stimulation start and end frames. These do not correspond to imaging frames! (optional: Note
            the start and end times.)
        4.  Change Time [m:s.ms] column to Time [s] with decimal dot.
        5.  Change all commas to dots. This changes e.g. 0,1234 to 0.1234, which matlab recognises as a number.
    """
    try:  # try opening the file with utf-8 encoding. This should work for all normal 2-photon imaging (without stimulation)
        with open(fname, "r", encoding="utf-8") as f:
            col_names, data = read_contents(f)
        with open(export_fname, "w", encoding=export_encoding) as f:
            f.write("\t".join(col_names) + "\n")
            for row in data:
                f.write("\t".join(list(map(lambda x: str(x), row))) + "\n")
    # if utf-8 did not work, try utf-16: this is most common in recordings with stimulation laser.
    except UnicodeDecodeError:
        print(f"utf-8 encoding did not work for {fname}. Trying utf-16.")

        try:
            with open(fname, "r", encoding="utf-16") as f:
                read_contents(f)
            with open(export_fname, "w", encoding=export_encoding) as f:
                f.write("\t".join(col_names) + "\n")
                for row in data:
                    f.write("\t".join(list(map(lambda x: str(x), row))) + "\n")
        except UnicodeDecodeError:
            print("utf-16 encoding did not work either.")
            raise Exception("Opening file failed!")
