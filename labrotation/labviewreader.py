NUM_COLS = 20  # text file has 20 columns for current version of labview Movementdetection.vi
from typing import List


class LabViewReader:
    """
     Structure of normal LabView txt output files (not XYtime.txt, but XY.txt):
    columns (indexing from 1):
    1: cumulative distance (?) -> Rounds in labview
    2: velocity -> Speed in labview
    3: smoother cumulative distance (?) -> Total distance
    4: distance on belt (resets with new round) -> Distance per round
    5: reflectivity -> Reflectivity
    6: constant -> Lick detection
    7: number of laps -> Stripes in total
    8: marks new lap (0 otherwise) -> Stripes in round
    9: time? -> Total time
    10: time since new lap started (i.e. resets every new lap) -> Time per round
    11-19: "Stimuli" (we don't use stimuli)
    20: pupil area
    """
    labview_data = []

    def __init__(self, absolute_path: str):
        self.labview_data = [[] for i in range(NUM_COLS)]
        with open(absolute_path) as f:
            for line in f:
                row = line.rstrip('\n').split('\t')
                for index, element in enumerate(row):
                    self.labview_data[index].append(element)
        # remove last line from data, which is only zeros.
        self._remove_incomplete_row()
        self._check_consistency()
        print(f"LabViewReader: Opened {absolute_path}.")

    # adjusts file and removes last row
    def _remove_incomplete_row(self) -> None:
        """
        Corrects LabView data artifact resulting from stopping during writing.
        This function assumes that the first column of the provided list has a last element that needs to be removed.
        The other columns are also brought to the same length as the first by removing the last element.
        The changes are in-place (i.e. data_list is changed)!
        """
        self.labview_data[0].pop()
        final_length = len(self.labview_data[0])
        for col_index in range(1, len(self.labview_data)):
            if len(self.labview_data[
                       col_index]) > final_length:  # only one non-complete row should exist in file. Otherwise use while()!
                self.labview_data[col_index].pop()

    def _check_consistency(self) -> bool:
        """
        Once a file is read and corrected for recording artifacts, this function determines if the column lengths match.
        :return: true if all columns have same length, false otherwise.
        """
        col_len = len(self.labview_data[0])
        for col_ind in range(len(self.labview_data)):
            if col_len != len(self.labview_data[col_ind]):  # one column does not match first column length!
                print(f"LabViewReader: Columns 0 and {col_ind} do not match in length after correction!")
                return False
        print(f"LabViewReader: Column lengths check was successful. All columns have {col_len} entries.")
        return True

    def rounds(self) -> List:
        return self.labview_data[0]

    def velocity(self) -> List:
        return self.labview_data[1]

    def total_distance(self) -> List:
        return self.labview_data[2]

    def distance_per_round(self) -> List:
        return self.labview_data[3]

    def reflectivity(self) -> List:
        return self.labview_data[4]

    def lick_detection(self) -> List:
        return self.labview_data[5]

    def stripes_in_total(self) -> List:
        return self.labview_data[6]

    def stripes_in_round(self) -> List:
        return self.labview_data[7]

    def total_time(self) -> List:
        return self.labview_data[8]

    def time_per_round(self) -> List:
        return self.labview_data[9]
