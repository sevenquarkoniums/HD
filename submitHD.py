#!/usr/bin/env python
WORKDIR = '/projectnb/peaclab-mon/yijia'
from subprocess import call

for windowSize in [1]:
    for downSample in [45]:#[2**x for x in range(7)]:
        for trainMethod in ['HDaddSimilar']:#['closest','addFilter','HDadd','HDaddSimilar']
            for dimension in [10000]:
                for seed in [0]:
                    for anomalyTrain in ['all']:#['bt','cg','CoMD','ft','kripke','lu','mg','miniAMR','miniGhost','miniMD','sp']:
                        if downSample < 5:
                            mem = 128
                        elif downSample in [5]:
                            mem = 64
                        elif downSample > 5 and downSample < 20:
                            mem = 30
                        elif downSample >= 20:
                            mem = 15
                        name = 'HD_window%d_downsample%d_trainWith%s_dim%d_seed%d_%s' % (windowSize, downSample, trainMethod, dimension, seed, anomalyTrain)
                        shfile = '%s/HDcomputing/sh/%s.sh' % (WORKDIR, name) 
                        outpath = '%s/HDcomputing/out/%s.out' % (WORKDIR, name)
                        fsh = open(shfile, 'w')
                        line = '#!/bin/bash -l\n'
                        line += 'module load anaconda3\n'
                        line += './HDcomputing.py %d %d %s %d %d %s' % (windowSize, downSample, trainMethod, dimension, seed, anomalyTrain)
                        fsh.write(line)
                        fsh.close()

                        command = 'chmod +x %s\n' % shfile
                        call(command, shell=True)
                        command = 'qsub -l mem_total=%dG -cwd -o %s -j y %s\n' % (mem, outpath, shfile)
                        call(command, shell=True)

