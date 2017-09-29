import os
from os.path import join
import numpy as n
import glob

plates = n.array(glob.glob("*.sh"))
plates.sort()

for plate in plates:
	os.system("qsub "+plate)
