import os
import sys

rootdir = sys.argv[1]
destdir = sys.argv[2]
#prefix = sys.argv[3]

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
                outputname = inputname + '.pbqp.s'
                existname = f + '.pbqp.s'
                if (existname in existnames):
                    continue
                
                #cmd = 'llc -regalloc=greedy ' + inputname + ' -o ' + outputname
                #os.system(cmd)
                #cmd = 'llvm-mca ' + outputname + ' -mcpu=btver2 > ' + outputname + '.greedy'
                #os.system(cmd)

                cmd = 'llc -regalloc=pbqp ' + inputname + ' -o ' + outputname 
                os.system(cmd)                                           
                #cmd = 'llvm-mca ' + outputname + ' -mcpu=btver2 > ' + outputname + '.pbqp'
                #os.system(cmd)  
