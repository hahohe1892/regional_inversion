import scipy.sparse as sps
from scipy.sparse import linalg
import numpy as np
from numpy import matlib
from copy import copy, deepcopy
from netCDF4 import Dataset as NC
from scipy.interpolate import griddata
from scipy import ndimage
from celluloid import Camera
import subprocess
import matplotlib.pyplot as plt
#import PISM
import os.path

#ctx = PISM.Context()

def identify_neighbors(n,m, nan_list, talks_to):
    nan_count = np.shape(nan_list)[0]
    talk_count = np.shape(talks_to)[0]
    nn = np.zeros((nan_count*talk_count,2), dtype='int')
    j=[0, nan_count]
    for i in range(talk_count):
        nn[j[0]:j[1],:]=nan_list[:,1:3]+matlib.repmat(talks_to[i], nan_count,1)
        j[0]+=nan_count
        j[1]+=nan_count
    L = np.logical_or(np.logical_or((nn[:,0]<0), (nn[:,0]>n)), np.logical_or((nn[:,1]<0), (nn[:,1]>m)))
    nn = nn[~L]
    neighbors_list = np.zeros((np.shape(nn)[0],3), dtype='int')
    neighbors_list[:,0] = np.ravel_multi_index((nn[:,0],nn[:,1]), (n, m))
    neighbors_list[:,1] = nn[:,0]
    neighbors_list[:,2] = nn[:,1]
    neighbors_list = np.unique(neighbors_list, axis=0)
    nrows, ncols = neighbors_list.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [neighbors_list.dtype]}

    neighbors_list_final = np.setdiff1d(neighbors_list.view(dtype), nan_list.view(dtype))
    neighbors_list_final = neighbors_list_final.view(neighbors_list.dtype).reshape(-1, ncols)
    return neighbors_list_final

def inpaint_nans(B_rec):
    n, m = np.shape(B_rec)
    A=B_rec.flatten()
    nm=n*m
    k = np.isnan(A)
    nan_list_unrolled = np.where(k)[0]
    known_list = np.where(~np.isnan(A))[0]
    nan_count = len(nan_list_unrolled)
    nr = np.argwhere(np.isnan(B_rec))[:,0]
    nc = np.argwhere(np.isnan(B_rec))[:,1]
    nan_list = np.zeros((nan_count,3), dtype='int')
    nan_list[:,0] = nan_list_unrolled
    nan_list[:,1] = nr
    nan_list[:,2] = nc

    talks_to = [[-1,0], [0,-1],[1,0], [0,1]]
    neighbors_list = identify_neighbors(n, m, nan_list, talks_to)
    all_list = np.concatenate((nan_list, neighbors_list), axis=0)
    L = np.where(np.logical_and(all_list[:,1]>0, all_list[:,1]<n))[0]
    nl=np.shape(L)[0]
    if nl>0:
        down = sps.csr_matrix((np.ones(nl), (all_list[L,0], all_list[L,0]-1)), shape=(nm,nm))
        middle = sps.csr_matrix((np.ones(nl)*-2, (all_list[L,0], all_list[L,0])), shape=(nm,nm))
        up = sps.csr_matrix((np.ones(nl), (all_list[L,0], all_list[L,0]+1)), shape=(nm,nm))
        fda = down+middle+up
    else:
        fda = sps.csr_matrix((np.zeros(nm), (np.arange(nm), np.arange(nm))))
    L = np.where(np.logical_and(all_list[:,2]>0, all_list[:,2]<m))[0]
    nl=np.shape(L)[0]
    if nl>0:
        down = sps.csr_matrix((np.ones(nl), (all_list[L,0], np.maximum(0, all_list[L,0]-m))), shape=(nm,nm))
        middle = sps.csr_matrix((np.ones(nl)*-2, (all_list[L,0], all_list[L,0])), shape=(nm,nm))
        up = sps.csr_matrix((np.ones(nl), (all_list[L,0], all_list[L,0]+m)), shape=(nm,nm))
        fda+=down+middle+up

    rhs = -fda[:,known_list]*A[known_list]
    k=np.argwhere(np.any(fda[:,nan_list[:,0]]))[:,0]

    B=copy(A)
    fda = fda[:,nan_list[:,0]]
    B[nan_list[:,0]] = sps.linalg.lsqr(fda[k], rhs[k])[0]
    B = np.reshape(B, (n,m))
    return B


def get_nc_data(file, var, time):
    ds = NC(file)
    avail_vars = [nc_var for nc_var in ds.variables]
    if var not in avail_vars:
        raise ValueError('variable not found; must be in {}'.format(avail_vars))
    elif time == ':':
        var_data = ds[var][:][:]
    elif var in ['x', 'y']:
        var_data = ds[var][:]
    else:
        var_data = ds[var][time][:]
    return var_data

def deghost(field):
    '''PISM puts a buffer of 2 cells around each field on each processor.
    When run in parallel, this means that 2 cells on each subdomain are 
    copies from the neighboring subdomain from a different process. 
    If we do not want to mess with fields that
    do not belong to our subdomain, we need to get rid of these fields'''
    return field[2:-2,2:-2]

def modify_buffer_vel(vel_field):
    vel_field[1,:] = vel_field[2,:]
    vel_field[-2,:] = vel_field[-3,:]
    vel_field[:,1] = vel_field[:,2]
    vel_field[:,-2] = vel_field[:,-3]
    vel_field = vel_field[1:-1,1:-1]
    return np.array(vel_field, copy = True)
    
def shift(field, u, v, mask, dx):
    
    '''
    shape_preserver = np.zeros_like(field)
    #field = deghost(field)
    #u = deghost(u)
    #v = deghost(v)
    #mask = deghost(mask)
    u = modify_buffer_vel(u)
    v = modify_buffer_vel(v)
    field = field[1:-1,1:-1]
    mask = mask[1:-1,1:-1]
    '''
    
    x_shift, y_shift = np.meshgrid(range(np.shape(field)[1]), range(np.shape(field)[0]))
    uv_mag = np.zeros_like(u)+mask
    uv_mag = uv_mag * np.sqrt(u**2+v**2)
    u_shift = np.zeros_like(u)
    v_shift = np.zeros_like(v)
    u_shift[uv_mag>0] = (u[uv_mag>0]/uv_mag[uv_mag>0])*dx
    v_shift[uv_mag>0] = (v[uv_mag>0]/uv_mag[uv_mag>0])*dx
    x_shift = x_shift+u_shift
    y_shift = y_shift+v_shift

    points=np.zeros((len(x_shift.flatten()),2))
    
    points[:,0] = x_shift.flatten()
    points[:,1]=y_shift.flatten()
    xi, yi = np.meshgrid(range(np.shape(field)[1]), range(np.shape(field)[0]))

    newgrid = griddata(points, field.flatten(), (xi.flatten(), yi.flatten()), 'linear').reshape(np.shape(u))
    #outgrid = np.copy(field)
    #outgrid = np.copy(shape_preserver)
    #outgrid[2:-2,2:-2] = newgrid[1:-1,1:-1]
    #outgrid[4:-4,4:-4] = newgrid[4:-4,4:-4]
    #newgrid[np.isnan(newgrid)] = field[np.isnan(newgrid)]
    return newgrid

def gauss_filter(U, sigma, truncate):

    V=U.copy()
    V[np.isnan(U)]=0
    VV=ndimage.gaussian_filter(V,sigma=sigma,truncate=truncate)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=ndimage.gaussian_filter(W,sigma=sigma,truncate=truncate)

    Z = np.zeros_like(U)
    Z[WW != 0]=VV[WW != 0]/WW[WW != 0]
    #Z=VV/WW
    return Z

#from IPython.display import display, Javascript
import time
import hashlib
import shelve
import os
import subprocess
                         
def save_and_commit(notebook_path, branch_name, nc_file, commit_message):
    
    current_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode('ascii').strip()
    if current_branch != branch_name:
        raise ValueError('not on correct branch')
        
    if (os.path.exists(nc_file) and os.path.exists('./models/') and os.path.exists('./data/')) == False:
        raise ValueError('nc_file or target folder does not exist')
    
    start_md5 = hashlib.md5(open(notebook_path,'rb').read()).hexdigest()
    display(Javascript('IPython.notebook.save_checkpoint();'))
    current_md5 = start_md5
        
    while start_md5 == current_md5:
        time.sleep(1)
        current_md5 = hashlib.md5(open(notebook_path,'rb').read()).hexdigest()
                
    stage = ["git", "add", "{}".format(notebook_path)]
    commit = ["git", "commit", "-m", commit_message]
    try:
        proc = subprocess.check_output(stage, stderr=subprocess.STDOUT)
        proc = subprocess.check_output(commit, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        raise ValueError('something went wrong')
        
    hashmark =  subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    save_model = ["cp", "{}".format(nc_file), "./models/{}_{}.nc".format(branch_name, hashmark)]
    proc = subprocess.check_output(save_model, stderr=subprocess.STDOUT)

    bk = shelve.open('./data/{}_{}.pkl'.format(branch_name, hashmark),'n')
    exceptions = ['NPI_50m_DEM', 'NPI_DEM', 'NPI_DEM_o', 'R2_50m_Vel', 'R2_50m_Vx', 'R2_50m_Vy', 'R2_Vel', 'R2_Vx', 'R2_Vy', 'Z_09', 'Z_14', 'Z_50m_09', 'Z_50m_14', 'Z_50m_20', 'Z_50m_90', 'bed', 'bed_mask', 'bed_mask_meta', 'bed_o', 'dhdt_0914', 'dhdt_50m_0914', 'mat_DEM_Vel', 'mat_RADAR', 'ocean_50m_mask', 'retreat_50m_mask','smb_50m_fit', 'smb_fit', 'smb_net_0914', 'smb_net_df', 'smb_x', 'smb_y', 'smb_z', 'smb_xyz_df', 'surf', 'surf_mask','surf_mask_meta', 'surf_o', 'thk', 'thk_mask', 'thk_mask_meta', 'thk_o', 'vel_0914', 'vel_50m_0914', 'vel_df', 'vel_x', 'vel_y', 'vel_z', 'vel_xyz_df', 'x_50m', 'y_50m']
    for k in sorted(globals()):
        if k in exceptions:
            continue
        if k.split('_')[0]=='':
            continue
        try:
            bk[k] = globals()[k]
        except Exception:
            print('{} was not saved'.format(k))
            pass
    bk.close()


def stagger(x):
    x_stag = np.zeros_like(x)
    x_stag[1:-1,1:-1] = 0.25*(x[1:-1,1:-1]+x[1:-1,0:-2]+x[0:-2,0:-2]+x[0:-2,1:-1])
    return x_stag

def x_deriv(x, res):
    dxdx = np.zeros_like(x, dtype='float')
    dxdx[0:-2,0:-2] = 0.5 * (x[1:-1,0:-2]-x[1:-1,1:-1] + x[0:-2,0:-2] - x[0:-2,1:-1])/res
    return dxdx
def y_deriv(y, res):
    dydy = np.zeros_like(y, dtype='float')
    dydy[0:-2,0:-2] = 0.5*(y[0:-2,1:-1]-y[1:-1,1:-1] + y[0:-2,0:-2] - y[1:-1,0:-2])/res
    return dydy

def calc_slope(field, res):
    field_stag = stagger(field)
    dhdx = x_deriv(field_stag, res)
    dhdy = y_deriv(field_stag, res)
    slope = np.sqrt(dhdx**2 + dhdy**2)#*np.sign(np.maximum(dhdx, dhdy))

    return slope

def correct_high_diffusivity(surf, bed, dt, max_steps_PISM, res, A, g=9.8, ice_density=900, R=0.12, return_mask = False):
    H = surf - bed
    slope = np.zeros_like(H)+1e-4
    slope[2:-2, 2:-2] = calc_slope(surf, res)[2:-2,2:-2] #ignore margins pixels of slope since they are very large
    slope = np.maximum(slope, 1e-4)
    #px, py = np.gradient(surf, res)
    #slope[2:-2,2:-2] = np.sqrt(px ** 2 + py ** 2)[2:-2,2:-2]
    secpera = 31556926.
    T = (2*A*(g*ice_density)**3)/5
    max_allowed_thk = ((((dt/max_steps_PISM)*secpera/(res**2)/R)**(-1))/(T*slope**2))**(1/5)
    #diff_allowed  = R/((dt/max_steps_PISM)*secpera/res**2)
    #max_allowed_thk = np.ones_like(H)*1e10
    #max_allowed_thk[2:-2,2:-2] = (diff_allowed/(diffusivity/H[2:-2,2:-2]**5))**(1/5)
    #max_allowed_thk[np.isnan(max_allowed_thk)] = 1e10
    #max_allowed_thk = np.minimum(1e10, max_allowed_thk)
    H_rec = np.minimum(H, max_allowed_thk)
    bed = surf - H_rec
    if return_mask:
        corrected_thk = np.zeros_like(H_rec)
        corrected_thk[H_rec == max_allowed_thk] = 1
        return bed, corrected_thk
    else:
        return bed

def calc_diffusivity(model, S_rec, B_rec):
    model.bed_deformation_model().bed_elevation().local_part()[:] = B_rec
    model.bed_deformation_model().bed_elevation().update_ghosts()

    # we also need to update this copy of bed elevation (for consistency)
    model.geometry().bed_elevation.local_part()[:] = B_rec
    model.geometry().bed_elevation.update_ghosts()

    # model.geometry() stores ice thickness
    model.geometry().ice_thickness.local_part()[:] = np.maximum(0, S_rec - B_rec)
    model.geometry().ice_thickness.update_ghosts()

    dt = PISM.util.convert(1e-5, "year", "second")
    model.run_to(ctx.time.current() + dt)

    H_min = ctx.config.get_number("geometry.ice_free_thickness_standard")
    model.geometry().ensure_consistency(H_min)
    diags = model.stress_balance().diagnostics().asdict()
    diffusivity = diags['diffusivity'].compute().local_part()

    return diffusivity
'''
surf = np.zeros((212,123))
surf[2:-2,2:-2] = get_nc_data('Kronebreen_output.nc', 'usurf', -1)
bed =  np.zeros((212,123))
bed[2:-2,2:-2] = get_nc_data('Kronebreen_output.nc', 'topg', -1)
diffusivity = get_nc_data('Kronebreen_output.nc', 'diffusivity', -1)
'''
def movie(field_series, step=1, file_name = 'animation.mp4', **kwargs):
    fig = plt.figure()
    camera = Camera(fig)
    for f in range(0, len(field_series), step):
        plt.pcolor(field_series[f], **kwargs)
        camera.snap()
    animation = camera.animate()
    animation.save(file_name)
    cmd = ['xdg-open', file_name]
    subprocess.call(cmd)

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

def plot_list(list, r, c, **kwargs):
    if r*c<len(list):
        raise ValueError('plot too small for this list')
    
    fig, ax = plt.subplots(r, c)
    j,k = (0,0)
    for i,element in enumerate(list):
        field = ax[j,k].pcolor(element, **kwargs)
        if i == 0:
           k+=1
        elif (i-j)%((j+1)*(c-1))!=0:
            k+=1
        else:
            k-=(c-1)
            j+=1
    return fig, ax
        

def get_nearest(x_ref,y_ref,reference, x_dat, y_dat):
    grid_temp = []
    for i in range(len(x_dat)):
        abslat = np.abs(x_ref-x_dat[i])
        abslon= np.abs(y_ref-y_dat[i])

        c = np.maximum(abslon,abslat)
        latlon_idx = np.argmin(c)
        grid_temp.append(reference.flat[latlon_idx])
    return grid_temp

def append_file_name_with_hash(file_names, hash):
    for file in file_names:
        if file in ['Kronebreen_input.nc', 'Kronebreen_initial_setup.nc', 'Kronebreen_build_output.nc']:
            print('Stopping: these files should probably not be changed')
            break
        elif os.path.isfile('{}_{}'.format(file, hash)):
            print('Stopping: file already exists')
            break
        else:
            file_name_pieces = file.split('.')[:-1] #remove file ending from name
            file_name = file_name_pieces[0]
            for piece in file_name_pieces[1:]:
                file_name += '.'
                file_name += piece
            file_ending = file.split('.')[-1] #save file type
            cmd = ['mv', file, '{}_{}.{}'.format(file_name, hash, file_ending)]
            subprocess.call(cmd)

from scipy import ndimage as nd

def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    #import numpy as np
    #import scipy.ndimage as nd

    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def create_buffer(data, buffer_mask, width):
    data_orig = deepcopy(data)
    for i in range(width):
        data[buffer_mask==0] = np.nan
        boundary_buffer_mask = buffer_mask==0
        k = np.ones((3,3),dtype=int)
        boundary = ndimage.binary_dilation(boundary_buffer_mask==0, k) & boundary_buffer_mask
        data[boundary==1] = fill(data)[boundary==1]
        buffer_mask[boundary==1] = 1
        data[buffer_mask==0] = data_orig[buffer_mask==0]
    return data


def dp(field1, field2, vmin = -2, vmax = 2, cmap='RdBu'):
    fig, ax = plt.subplots()
    field = ax.imshow(field1 - field2, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(field, ax=ax)
    plt.show()


def pl(field, **kwargs):
    fig, ax = plt.subplots()
    f = ax.imshow(field, **kwargs)
    fig.colorbar(f, ax = ax)
    plt.show()


def pl3d(X, Y, Z, mask, **kwargs):
    fig, ax = plt.subplots()
    x, y = np.meshgrid(X, Y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, Z, cmap='viridis')
    mask_nan = np.ones_like(mask) * np.nan
    mask_nan[mask == 1] = Z[mask == 1] + 10
    ax.plot_surface(x, y, mask_nan, color = 'r')
    plt.show()


def qsplot(x,y,ux,uv,flux = None, thk = None, show = True, **kwargs):
    if flux is None:
        flux = np.ones_like(ux)
    if hasattr(ux, 'attrs'):
        if '_FillValue' in ux.attrs:
            ux.data[ux.data == ux.attrs['_FillValue']]  = np.nan
            uv.data[uv.data == uv.attrs['_FillValue']]  = np.nan
    thk_kwargs = {}
    if 'vmin_t' in kwargs:
        thk_kwargs['vmin'] = kwargs['vmin_t']
        del kwargs['vmin_t']
    if 'vmax_t' in kwargs:
        thk_kwargs['vmax'] = kwargs['vmax_t']
        del kwargs['vmax_t']

    if x[0] > x[-1]:
        x = np.flip(x)
        ux = -1 * ux
    if y[0] > y[-1]:
        y = np.flip(y)
        #uv = -1 * uv
    if thk is not None:
        plt.pcolor(x,y,thk, cmap = 'twilight', **thk_kwargs)
    plt.streamplot(x, y, ux, uv, **kwargs)
    plt.quiver(x, y, ux, uv, flux, cmap = 'jet')
    if show == True:
        plt.show()


def movieqs(extra, step=1, file_name = 'animation.mp4', **kwargs):
    try:
        ux = extra[1]['uvelsurf']
        uv = extra[1]['vvelsurf']
        if hasattr(ux, 'attrs'):
            if '_FillValue' in ux.attrs:
                ux.data[ux.data == ux.attrs['_FillValue']]  = np.nan
                uv.data[uv.data == uv.attrs['_FillValue']]  = np.nan

        x = extra[1].x.data
        y = extra[1].y.data
        flux = extra[1]['flux_mag']
        thk = extra[1]['thk']
        usurf = extra[1]['usurf']
    except(KeyError):
        ux = extra['uvelsurf']
        uv = extra['vvelsurf']
        x = extra.x.data
        y = extra.y.data
        flux = extra['flux_mag']
        thk = extra['thk']
        usurf = extra['usurf']
    max_thk = np.max(thk.data)
    start_points_x = np.meshgrid(x,y)[0][~np.isnan(ux[0].data)]
    start_points_y = np.meshgrid(x,np.flip(y))[1][~np.isnan(ux[0].data)]

    fig = plt.figure()
    camera = Camera(fig)
    for i in range(0, len(ux), step):
        dy, dx = np.gradient(usurf[i]*ux[i]/ux[i])
        #qsplot(x,y,-dx, dy, flux[i], thk[i], show = False, density = 3, color = 'w', linewidth = .3, vmax_t = max_thk, broken_streamlines = False, arrowsize = 1e-10, start_points = np.array([start_points_x, start_points_y]).T)
        qsplot(x,y,ux[i], uv[i], flux[i], thk[i], show = False, color = 'w', linewidth = .1, vmax_t = max_thk, broken_streamlines = False, arrowsize = 1e-10, start_points = np.array([start_points_x, start_points_y]).T)
        camera.snap()
    animation = camera.animate()
    animation.save(file_name, dpi = 300)
    cmd = ['xdg-open', file_name]
    subprocess.call(cmd)


def normalize(x):
    return (x-np.nanmin(x))/(np.nanmax(x) - np.nanmin(x))

def do_kdtree(new_bed, mask, buffer):
    remove_ind = (buffer.numpy() == 1).flatten() + (mask.numpy() == 0).flatten()
    x,y = np.meshgrid(range(new_bed.shape[1]), range(new_bed.shape[0]))
    xy_arr = np.dstack([x.flatten(), y.flatten()])[0]
    xy_arr = xy_arr[~remove_ind]
    points = np.flip(np.array(np.where(buffer.numpy() == 1)), axis = 0).transpose()
    mytree = scipy.spatial.cKDTree(xy_arr)
    dist, indexes = mytree.query(points)
    xy_ind = xy_arr[indexes]
    bed_values = [new_bed[i[1], i[0]] for i in xy_ind]
    new_bed[buffer.numpy() == 1] = bed_values
    return new_bed
