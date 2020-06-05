import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import casa_cube as casa
import os

source_name = 'HD142527'
isotope = ['12CO','13CO','C18O']
iso = 1
directory = f'{source_name}_slices'
if not os.path.exists(directory):
    os.mkdir(directory)

'''
continuum = casa.Cube(f'{source_name}_continuum.fits')
im = np.nan_to_num(continuum.image[0,:,:])*1000
#im = continuum._Jybeam_to_Tb(im)

plt.figure()
plt.clf()
plt.xlabel('pixel')
plt.ylabel('pixel')
image = plt.imshow(im, origin='lower', cmap=cm.hot, vmin=0, vmax=np.max(im))
cbar = plt.colorbar(image)
cbar.set_label('mJy/beam')

beam_maj = continuum.bmaj/continuum.pixelscale
beam_min = continuum.bmin/continuum.pixelscale
beam_bpa = continuum.bpa
ax = plt.gca()
beam = Ellipse(xy=(40,40), width=beam_min, height=beam_maj, angle=-beam_bpa, fill=True, color='black')
ax.add_patch(beam)

plt.savefig(f'{directory}/{source_name}_continuum.jpg', quality=95)#, dpi=1200)
plt.show()
'''
'''
for i in range(iso):
    
    source = casa.Cube(f'{source_name}_{isotope[i]}_contsub.fits')
    cube = np.nan_to_num(source.image[:,:,:])*1000
    #cube = source._Jybeam_to_Tb(cube)

    nv = source.nv
    for j in range(nv):
        iv = j 
        im = np.nan_to_num(source.image[iv,:,:])*1000.

        plt.figure()
        plt.clf()
        plt.xlabel('pixel')
        plt.ylabel('pixel')
        image = plt.imshow(im, origin='lower', cmap=cm.Purples_r, vmin=0, vmax=np.max(cube))
        cbar = plt.colorbar(image)
        cbar.set_label('mJy/beam')

        beam_maj = source.bmaj/source.pixelscale
        beam_min = source.bmin/source.pixelscale
        beam_bpa = source.bpa
        ax = plt.gca()
        beam = Ellipse(xy=(40,40), width=beam_min, height=beam_maj, angle=-beam_bpa, fill=True, color='black')
        ax.add_patch(beam)
        
        plt.savefig(f'{directory}/{source_name}_{isotope[i]}_contsub_channel_{j+1}.jpg', quality=95)#, dpi=1200)

        plt.close()
'''
'''
np.set_printoptions(threshold=np.inf) 
for i in range(iso):
    
    source = casa.Cube(f'{source_name}_{isotope[i]}_contsub.fits')

    fig, ax = plt.subplots(nrows=4, ncols=5, sharex=True, sharey=True, squeeze=False)
    
    iv = 0*20  #set of 20 maps. to chage to next set, change 0*20 to 1*20 for example. change name of savefile as well. for HD142527 there are only 110 channels, so go up to five sets (100 channels) otherwise there will be an error.
    cube = np.nan_to_num(source.image[:,:,:])*1000
    #cube = source._Jybeam_to_Tb(cube)

    im = np.nan_to_num(source.image[iv+0,:,:])*1000
    #im = source._Jybeam_to_Tb(im)

    ax[0,0].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[0,0].text(50,300, f"v={source.velocity[iv+0]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+1,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[0,1].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[0,1].text(50,300, f"v={source.velocity[iv+1]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+2,:,:])*1000
    #im = source._Jybeam_to_Tb(im)    
    ax[0,2].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[0,2].text(50,300, f"v={source.velocity[iv+2]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+3,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[0,3].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[0,3].text(50,300, f"v={source.velocity[iv+3]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+4,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[0,4].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[0,4].text(50,300, f"v={source.velocity[iv+4]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+5,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[1,0].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[1,0].text(50,300, f"v={source.velocity[iv+5]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+6,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[1,1].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[1,1].text(50,300, f"v={source.velocity[iv+6]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+7,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[1,2].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[1,2].text(50,300, f"v={source.velocity[iv+7]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+8,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[1,3].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[1,3].text(50,300, f"v={source.velocity[iv+8]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+9,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[1,4].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[1,4].text(50,300, f"v={source.velocity[iv+9]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+10,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[2,0].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[2,0].text(50,300, f"v={source.velocity[iv+10]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+11,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[2,1].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[2,1].text(50,300, f"v={source.velocity[iv+11]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+12,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[2,2].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[2,2].text(50,300, f"v={source.velocity[iv+12]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+13,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[2,3].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[2,3].text(50,300, f"v={source.velocity[iv+13]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+14,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[2,4].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[2,4].text(50,300, f"v={source.velocity[iv+14]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+15,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[3,0].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[3,0].text(50,300, f"v={source.velocity[iv+15]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+16,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[3,1].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[3,1].text(50,300, f"v={source.velocity[iv+16]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+17,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[3,2].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[3,2].text(50,300, f"v={source.velocity[iv+17]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+18,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    ax[3,3].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[3,3].text(50,300, f"v={source.velocity[iv+18]:.2f} km/s", fontsize=6)

    im = np.nan_to_num(source.image[iv+19,:,:])*1000
    #im = source._Jybeam_to_Tb(im)
    image = ax[3,4].imshow(im, origin='lower', cmap=cm.Purples, vmin=0, vmax=np.max(cube))
    ax[3,4].text(50,300, f"v={source.velocity[iv+19]:.2f} km/s", fontsize=6)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(image, cax=cbar_ax)
    cbar.set_label('mJy/beam')

    fig.text(0.5, 0.04, 'pixel', va='center')
    fig.text(0.04, 0.5, 'pixel', va='center', rotation='vertical')

    plt.savefig(f'{directory}/{source_name}_{isotope[i]}_contsub_set1.pdf')
    #plt.show()
'''
