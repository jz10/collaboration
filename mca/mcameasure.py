import os
import sys

rootdir = sys.argv[1]
destdir = sys.argv[2]
prefix = sys.argv[3]

currnames = []
for dirpath, dirs, files in os.walk(destdir):
    for f in files:
        #filename = dirpath + '/' + f
        currnames.append(f)
        
existnames = set(currnames)

for dirpath, dirs, files in os.walk(rootdir):
    path = dirpath.split('/')
    outname = path[len(path) - 1]
    if (outname in existnames):
        continue

    for f in files:
        if f[0] == prefix:
            filenames = f.split('.')
            if len(filenames) >= 2:
                if filenames[len(filenames) - 1] == 's':
                    inputname = dirpath + '/' + f
                    existname = f[:len(f) - 2] + '.res'
                    outputname = destdir + '/' + existname 
                    if (existname in existnames):
                        continue
                    
                    cmd = 'llvm-mca ' + inputname + ' -mcpu=btver2 > ' + outputname + ' 2>&1 '
                    # print(cmd)
                    os.system(cmd)
