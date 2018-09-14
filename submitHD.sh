#!/usr/bin/env python

from subprocess import call

for windowSize in range(1, 11):
    for downSample in [2**x for x in range(7)]:
        for trainMethod in ['closest','addFilter','HDadd']:
            for dimension in [10000]:
                for seed in [0]:
                    if downSample == 64:
                        mem = 64
                    shfile = '/projectnb/peaclab-mon/yijia/HDcomputing/sh/HD_window%d_downsample%d_trainWith%s_dim%d_seed%d.sh' % (windowSize, 
                                                                downSample, trainMethod, dimension, seed)
                    fsh = open(shfile, 'w')
                    line = '#!/bin/bash -l\n'
                    line += 'module load anaconda3\n'
                    line += './HDcomputing.py %d %d %s %d %d' % (windowSize, downSample, trainMethod, dimension, seed)
                    fsh.write(line)
                    fsh.close()

                    command = 'chmod +x %s\n' % shfile
                    call(command, shell=True)
                    outpath = '/projectnb/peaclab-mon/yijia/HDcomputing/out/HD_window%d_downsample%d_trainWith%s_dim%d_seed%d.out' % (windowSize,
                                                                                    downSample, trainMethod, dimension, seed)
                    command = 'qsub -l mem_total=%dG -cwd -o %s -j y %s\n' % (mem, outpath, shfile)
                    call(command, shell=True)

