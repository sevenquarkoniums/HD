#!/usr/bin/env python

from subprocess import call
metrics = []
with open('/projectnb/peaclab-mon/yijia/HD/allMetrics.txt', 'r') as f:
    for line in f:
        metrics.append(line[:-1])

for metric in ['Active(anon)_meminfo']:#metrics:
    for feat in ['std','mean']:
        mem = 8
        shfile = '/projectnb/peaclab-mon/yijia/HDcomputing/sh/drawMetric_%s_%s.sh' % (metric, feat)
        fsh = open(shfile, 'w')
        line = '#!/bin/bash -l\n'
        line += 'module load anaconda3\n'
        line += './drawMetric.py %s %s' % (metric, feat)
        fsh.write(line)
        fsh.close()

        command = 'chmod +x %s\n' % shfile
        call(command, shell=True)
        outpath = '/projectnb/peaclab-mon/yijia/HDcomputing/out/drawMetric_%s_%s.out' % (metric, feat)
        command = 'qsub -l mem_total=%dG -cwd -o %s -j y %s\n' % (mem, outpath, shfile)
        call(command, shell=True)
