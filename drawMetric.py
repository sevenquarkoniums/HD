#!/usr/bin/env python
"""


run by:


### TODO ###


### warning ###


"""
import sys
#=========================
env = 'scc'
small = False
metric = 'Active(file)_meminfo' if env == 'dell' else sys.argv[1]# dcopy: Active(anon)_meminfo, leak: SMSG_nrx_cray_aries_r, linkclog: user_procstat.
feat = 'std' if env == 'dell' else sys.argv[2]
sliding = False

dataFolder = '/projectnb/peaclab-mon/yijia/data/volta' if env == 'scc' else 'C:/Programming/monitoring/data'
marginCut = 60
apps = ['bt','cg'] if small else ['bt','cg','CoMD','ft','kripke','lu','mg','miniAMR','miniGhost','miniMD','sp']
appInputs = ['X'] if small else ['X','Y','Z']
types = ['dcopy','leak','linkclog','dial','memeater','none']
epochLength = 45
#=========================
import datetime
now = datetime.datetime.now()

import pandas as pd
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg') # for running in linux batch job.
import matplotlib.patches as mpatches

def getfiles(path):
    # get all the files with full path.
    fileList = []
    for root, directories, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            fileList.append(filepath)
    return fileList


#================================
# main function starts.

features = {}
for app in apps:
    print()
    print(app)
    features[app] = {}
    for itype in types:
        print(itype)
        features[app][itype] = []
        for appInput in appInputs:
            if itype != 'none':
                files = getfiles('%s/%s_%s/%s_100' % (dataFolder, app, appInput, itype))
                nodes = [x for x in files if int(x[-5])==1]# only anomalous.
            else:
                nodes = getfiles('%s/%s_%s/%s_100' % (dataFolder, app, appInput, itype))
                for jtype in ['dcopy','leak','linkclog','dial','memeater']:
                    files = getfiles('%s/%s_%s/%s_100' % (dataFolder, app, appInput, jtype))
                    nodes += [x for x in files if int(x[-5])!=1]
            for inode in nodes:
                df = pd.read_csv(inode).iloc[marginCut:-marginCut]
                if sliding:
                    df_rolling = df[[metric]].rolling(epochLength)
                    if feat == 'std':
                        feature = df_rolling.std()[epochLength - 1:][metric]
                    elif feat == 'mean':
                        feature = df_rolling.mean()[epochLength - 1:][metric]
                else:
                    if feat == 'std':
                        feature = [df[metric].std()]
                    elif feat == 'mean':
                        feature = [df[metric].mean()]
                features[app][itype] += list(feature)

print('drawing..')
fs = 20
plt.rc('xtick', labelsize=fs)
plt.rc('ytick', labelsize=fs)
fig, ax = plt.subplots(figsize=(10,8))
for iapp, app in enumerate(apps):
    xs = [0.2+iapp+0.6/len(features[app]['none'])*inode for inode in range(len(features[app]['none']))]
    ax.plot(xs, features[app]['none'], 'xk')
    labels = ['xr','xb','xy','xg','xc']
    for idx, itype in enumerate(['dcopy','leak','linkclog','dial','memeater']):
        xs = [0.2+iapp+0.6/len(features[app][itype])*inode for inode in range(len(features[app][itype]))]
        ax.plot(xs, features[app][itype], labels[idx])
ax.set_xlabel('Applications', fontsize=fs)
ax.set_ylabel(metric, fontsize=fs)
ax.set_xlim(0, 11)
#ax.set_ylim(0, 1)
plt.xticks([0.5+x for x in range(11)], apps, rotation=45)
ymin,ymax = ax.get_ylim()
## background.
for start in range(0, 11, 2):
    ax.add_patch(
    mpatches.Rectangle(
        (start, ymin),   # (x,y)
        1,          # width
        ymax - ymin,          # height
        alpha = 0.2,
        facecolor = 'black',
        edgecolor = 'black'
    ))
## legend.
labels = ['healthy','dcopy','leak','linkclog','dial','memeater']
patches = []
patches.append(mpatches.Patch(color='k', alpha=1, label=''))
patches.append(mpatches.Patch(color='r', alpha=1, label=''))
patches.append(mpatches.Patch(color='b', alpha=1, label=''))
patches.append(mpatches.Patch(color='y', alpha=1, label=''))
patches.append(mpatches.Patch(color='g', alpha=1, label=''))
patches.append(mpatches.Patch(color='c', alpha=1, label=''))
lgd = ax.legend(handles=patches, labels=labels, fontsize=fs, ncol=3, loc='lower left', bbox_to_anchor=(0.1, 1))

plt.tight_layout()
if env == 'dell':
    plt.savefig('C:/Programming/monitoring/results/%s_%s_%s_%s.png' % ('all', metric, feat, 'sli' if sliding else 'entire'), 
                bbox_extra_artists=(lgd,), bbox_inches='tight')
elif env == 'scc':
    plt.savefig('/projectnb/peaclab-mon/yijia/results/drawMetric/%s_%s_%s_%s.png' % ('all', metric, feat, 'sli' if sliding else 'entire'), 
                bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()




print('finished in %d seconds.' % (datetime.datetime.now()-now).seconds)

