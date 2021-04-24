import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import cv2, time
from osgeo import gdal, gdalnumeric, ogr, osr

from OpenGL.GL import *
from OpenGL.GLUT import *

try:
    DEFAULT_RESAMPLE = gdal.GRIORA_Bilinear
    CAN_SPECIFY_RESAMPLE = True
except:
    DEFAULT_RESAMPLE = None
    CAN_SPECIFY_RESAMPLE = False
def readRgb(dset, x,y, w,h, srcw=None,srch=None, resample=DEFAULT_RESAMPLE):
    if srcw is None:
        srcw,srch = w,h
    # Certain versions of GDAL do not support resample_alg.
    if CAN_SPECIFY_RESAMPLE:
        cbuf = dset.ReadRaster(x,y, srcw,srch,w,h, resample_alg=resample)
    else:
        cbuf = dset.ReadRaster(x,y, srcw,srch,w,h)
    return np.transpose(np.frombuffer(cbuf, dtype=np.uint8).reshape(3,h,w), [1,2,0])

def get_tiff_patch(tiffName, patchBbox, res):
    dset = gdal.Open(tiffName)
    gt = dset.GetGeoTransform()
    xform_native2pix = np.array([1.0/gt[1],gt[2],-gt[0]/gt[1],gt[4],1.0/gt[5],-gt[3]/gt[5]]).reshape(2,3) # 2x3, we use 1d vector
    x,y = (xform_native2pix @ ((patchBbox[0], patchBbox[1], 1.)))[:2]
    w,h = (xform_native2pix @ ((patchBbox[0]+patchBbox[2], patchBbox[1]+patchBbox[3], 1.)))[:2] - (x,y)
    x,y,w,h = int(x),int(y),int(w),int(h)
    if w < 0: x,w = x+w,-w
    if h < 0: y,h = y+h,-h
    ww = int(res)
    hh = int(res*(h/w))
    img = readRgb(dset, x,y,ww,hh, w,h)
    return img


def get_dc_lidar(cfg):
    import pylas, ctypes
    stride = 1
    #f = '/data/lidar/USGS_LPC_VA_Fairfax_County_2018_e1617n1924.laz'
    f,stride = '/data/lidar/dc1.las', cfg.get('stride',8)
    #f,stride = '/data/lidar/PA_Statewide_S_2006-2008_002771.las', 2
    #f,stride = '/data/lidar/airport.las', 4

    cfg.setdefault('maxDepth', 8)

    st0 = time.time()
    with pylas.open(f) as fh:
        N = -1
        st1 = time.time()
        las = fh.read()
        print('   - las read     took {:.2f}ms ({} pts)'.format((time.time()-st1)*1000,len(las.x)))

        st1 = time.time()
        x,y,z = las.x[0:N:stride],las.y[:N:stride],las.z[:N:stride]
        inten,angle = las.intensity[0:N:stride], las.scan_angle_rank[:N:stride]
        print('   - las slice    took {:.2f}ms ({} pts)'.format((time.time()-st1)*1000, len(x)))

        st1 = time.time()
        qq = cfg.get('qq',.05)
        qqz0, qqz1 = .0000001, .99995 # Because many high z's are outliers, reject points >quantile(qqz1)
        (x1,x2),(y1,y2),(z1,z2) = np.quantile(x[::2],[qq,1-qq]), np.quantile(y[::2],[qq,1-qq]), np.quantile(z[::2],[qqz0,qqz1])
        print('   - las quantile took {:.2f}ms'.format((time.time()-st1)*1000))

        maxEdge = max(x2-x1, y2-y1)
        #xx,yy,zz = (x-x1) / maxEdge, (y-y1) / maxEdge, (z-z1) / maxEdge

        st1 = time.time()
        #pts = np.stack((xx,yy,zz), -1).astype(np.float32)
        pts = ((np.stack((x,y,z), -1) - (x1,y1,z1)) / maxEdge).astype(np.float32)
        inten = inten[(pts>0).all(1) & (pts<1).all(1) & (pts[:,2]<(z2-z1)/maxEdge)]
        angle = angle[(pts>0).all(1) & (pts<1).all(1) & (pts[:,2]<(z2-z1)/maxEdge)]
        pts = pts[(pts>0).all(1) & (pts<1).all(1) & (pts[:,2]<(z2-z1)/maxEdge)]
        print('   - las stack    took {:.2f}ms'.format((time.time()-st1)*1000))
        size = 2
        print(pts[::pts.shape[0]//5])
    print(' - las total load took {:.2f}ms'.format((time.time()-st0)*1000))

    vals = np.ones_like(pts[:,0])

    M = np.eye(4, dtype=np.float32)
    M[:3, 3] = (x1,y1,z1)
    M[:3,:3] = np.diag((maxEdge,)*3)

    endPoints = np.array(( (0,0,0,1.),
                           (1,1,1,1.))) @ M.T
    endPoints = endPoints[:, :3] / endPoints[:, 3:]
    utm_bbox = *endPoints[0,:2], *(endPoints[1,:2]-endPoints[0,:2])
    ox,oy = -2, -8 # Tiff or point-cloud is mis-aligned!
    ox,oy = 0,0
    utm_bbox = (utm_bbox[0]+ox,utm_bbox[1]+oy,utm_bbox[2],utm_bbox[3])
    img = get_tiff_patch('/data/dc_tiffs/dc.tif', utm_bbox, 2048*2)
    #img = get_tiff_patch('/data/dc_tiffs/dupont1/whole.32618.tif', utm_bbox, 2048*2)

    return dict(
            pts=pts,
            inten=inten,
            angle=angle,
            vals=vals,
            grid2native=M,
            img=img,
            maxEdge=maxEdge,
            pix2meter=maxEdge
            )
