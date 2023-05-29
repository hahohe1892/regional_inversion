from matplotlib.colors import LightSource
import numpy as np
import matplotlib.pyplot as plt


def hillshade(z, overlay, mask, res, vert_exag = 10, mode = 'hsv', azdeg = 315):
    ls = LightSource(azdeg, altdeg=45)
    cmap = plt.cm.gist_earth

    #fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(8, 9))
    #plt.setp(axs.flat, xticks=[], yticks=[])

    # Vary vertical exaggeration and blend mode and plot all combinations
    # Show the hillshade intensity image in the first row
    plt.imshow(ls.hillshade(z, vert_exag=vert_exag, dx=res, dy=res), cmap='gray')

    # Place hillshaded plots with different blend modes in the rest of the rows
    #for ax, mode in zip(col[1:], ['hsv', 'overlay', 'soft']):
    rgb = ls.shade(overlay, cmap=cmap, blend_mode=mode,
                           vert_exag=vert_exag, dx=res, dy=res)
    plt.imshow(rgb, alpha = .5)

    plt.show()
    
