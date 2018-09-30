import os
import sys

def GetSeqs(seqname):
    funcNames = []
    funcSeqs = []
    lines = list(open(seqname))
    for line in lines:
        elems = line.split(' ')
        if (len(elems) > 2):
            funcName = elems[0]
            if (elems[1] == '--'):
                funcNames.append(funcName)
                seqs = line.split('--')
                funcSeqs.append(seqs[1])

    return funcNames, funcSeqs

def GetIPCValues(mcaname):
    funcNames =[]
    ipcs = []
    lines = list(open(mcaname))
    for line in lines:
        elems = line.split(' ')
        if (elems[0] == 'Function' and elems[1] == 'Name:'):
            # Add function names
            funcNames.append(elems[2])
        if (elems[0] == 'IPC:'):
            # Add IPC value
            values = line.split(':')
            strVal = values[1]
            try: 
                ipcs.append(strVal)
            except (ValueError):
                print('Wrong IPC value: ' + strVal)

    if (len(funcNames) != len(ipcs)):
        funcNames[:] = []
        ipcs[:] = []
        
    return funcNames, ipcs
            
def GetIPCLabel(filename, seqname, mcaname_greedy, mcaname_pbqp, outputname):
    try:
        seqnames, seqs = GetSeqs(seqname)
        greedynames, greedyipcs = GetIPCValues(mcaname_greedy)
        pbqpnames, pbqpipcs = GetIPCValues(mcaname_pbqp)
        # Write to output file
        if (len(seqnames) == len(greedynames) and len(seqnames) == len(pbqpnames)):
            with open(outputname, "w") as text_file:
                for i in range(0, len(seqnames)):
                    label = 'greedy'
                    if (greedyipcs[i] > pbqpipcs[i]):
                        label = 'pbqp'
                    res = filename + ' ' + seqnames[i] + ' ' + label + ' --' + seqs[i]
                    print(res, file = text_file)
                        
    except (OSError, IOError):
        print("Error file: " + seqname)
    
rootdir = sys.argv[1]
mcadir = sys.argv[2]
seqdir = sys.argv[3]
destdir = sys.argv[4]

currnames = []
for dirpath, dirs, files in os.walk(destdir):
    for f in files:
        currnames.append(f)

existnames = set(currnames)

for dirpath, dirs, files in os.walk(rootdir):
    for f in files:
        print(f)
        filenames = f.split('.')
        if len(filenames) >= 2:
            if filenames[len(filenames) - 1] == 'bc':
                inputname = dirpath + '/' + f
                mcaname_greedy = mcadir + '/' + f + '.res'
                mcaname_pbqp = mcadir + '/' + f + '.pbqp.res'
                seqname = seqdir + f + '.seq'
                existname = f + '.label'
                outputname = destdir + '/' + existname
                if (existname in existnames):
                    continue

                GetIPCLabel(f, seqname, mcaname_greedy, mcaname_pbqp, outputname)
