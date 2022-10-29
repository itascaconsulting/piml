import itasca as it
import numpy as np

class _gpa:
    def disp(self):
        return np.array(tuple(gp.disp() for gp in it.gridpoint.list()))
gpa = _gpa()

