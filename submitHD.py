#!/usr/bin/env python

from subprocess import call

for windowSize in [2]:
    for downSample in [50]:#[2**x for x in range(7)]:
        for trainMethod in ['closest']:#,'addFilter','HDadd']:
            for dimension in [10000]:
                for seed in [0]:
                    for selectApp in ['bt']:#['all','bt','cg','CoMD','ft','kripke','lu','mg','miniAMR','miniGhost','miniMD','sp']:
                        if downSample == 1:
                            mem = 128
                        elif downSample in [5]:
                            mem = 64
                        elif downSample > 5:
                            mem = 32
                        name = 'HD_window%d_downsample%d_trainWith%s_dim%d_seed%d_%s' % (windowSize, downSample, trainMethod, dimension, seed, selectApp)
                        shfile = '/projectnb/peaclab-mon/yijia/HDcomputing/sh/%s.sh' % (name) 
                        outpath = '/projectnb/peaclab-mon/yijia/HDcomputing/out/%s.out' % (name)
                        fsh = open(shfile, 'w')
                        line = '#!/bin/bash -l\n'
                        line += 'module load anaconda3\n'
                        line += './HDcomputing.py %d %d %s %d %d %s' % (windowSize, downSample, trainMethod, dimension, seed, selectApp)
                        fsh.write(line)
                        fsh.close()

                        command = 'chmod +x %s\n' % shfile
                        call(command, shell=True)
                        command = 'qsub -l mem_total=%dG -cwd -o %s -j y %s\n' % (mem, outpath, shfile)
                        call(command, shell=True)

