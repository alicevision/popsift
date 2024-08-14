import sys, string, os
import subprocess

test_image_dirs = [ 'ubc', 'bark', 'boat', 'graf', 'wall', 'bikes' 'trees', 'leuven', ]

test_images = [ 'img1.ppm', 'img2.ppm', 'img3.ppm', 'img4.ppm', 'img5.ppm', 'img6.ppm' ]

executable = sys.argv[1]
dataset    = sys.argv[2]

print ("Running command {}".format(executable))
print ("Dataset location {}".format(dataset))

for d in test_image_dirs:
    for i in test_images:
        img = dataset + '/' + d + '/' + i
        if(os.path.isfile(img)):
            print ("Calling {} -i {}".format(executable,d+i))
            result = subprocess.run([executable, "-i",img])
            if(result.returncode==0):
                print ("Test ran successfully")
            else:
                print ("Test failed")
                sys.exit(-1)

sys.exit(0)
    
        # output_features = open('output-features.txt')
        # new_features = sorted(output_features.readlines())

        # ref_path     = os.path.dirname(os.path.realpath(__file__)) + '/features/popsift-default-big_set/' + img + '.feat'
        # ref_features = open(ref_path)
        # old_fatures  = sorted(ref_features.readlines())

