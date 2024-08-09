import sys, string, os
import subprocess

executable = sys.argv[1]
image      = sys.argv[2]

print ("Running command {}".format(executable))
print ("on image {}".format(image))

subprocess.check_call([executable, "-i",image])

