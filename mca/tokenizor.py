import os
import sys

rootdir = sys.argv[1]
destdir = sys.argv[2]

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
        filenames = f.split('.')
        if len(filenames) >= 2:
            if filenames[len(filenames) - 1] == 'bc':
                inputname = dirpath + '/' + f
                outputname = destdir + '/' + f + '.seq'
                
                if (outputname in existnames):
                    continue
                
                cmd = 'opt -analyze -tokenizer ' + inputname + ' > ' + outputname + ' 2>&1 '
                os.system(cmd)
