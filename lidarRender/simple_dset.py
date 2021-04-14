import gdal
import numpy as np, cv2

# Small wrapper around gdal.Dataset
# Implements some functions that correspond to tview.GeoDataset ones.
# Tries to handle partial out-of-bounds box accesses.
class SimpleGeoDataset:
    def __init__(self, path):
        self.dset = gdal.Open(path)

        self.pix2native_ = np.array(self.dset.GetGeoTransform())
        self.pix2native_ = self.pix2native_[[1,2,0, 4,5,3]].reshape(2,3)
        self.native2pix_ = np.linalg.inv(np.vstack((self.pix2native_,np.array((0,0,1.)).reshape(1,3))))[:2]

    def native2pix(self, x, y):
        return (self.native2pix_ @ np.array((x,y,1.))).squeeze()
    def pix2native(self, x, y):
        return (self.pix2native_ @ np.array((x,y,1.))).squeeze()

    def getProjection(self):
        return self.dset.GetProjectionRef()

    def getRawSizeX(self): return self.dset.RasterXSize
    def getRawSizeY(self): return self.dset.RasterYSize

    def bboxNative(self, bb_n, outw,outh, unused=False):
        if isinstance(bb_n, (tuple,list)): bb_n = np.array(bb_n)
        tl = self.native2pix(*bb_n[:2])
        wh = self.native2pix(*(bb_n[:2]+bb_n[2:])) - tl
        if wh[0] < 0: tl[0],wh[0] = tl[0] + wh[0], -wh[0]
        if wh[1] < 0: tl[1],wh[1] = tl[1] + wh[1], -wh[1]

        print(' - accessing pix', tl,tl+wh)
        return self.bboxPix((tl[0],tl[1],wh[0],wh[1]), outw,outh)


    def bboxPix(self, tlwh, outh,outw, unused=False, nir=False):
        tl,wh = tlwh[:2], tlwh[2:]
        H,W = self.getRawSizeY(), self.getRawSizeX()

        # Try to fix invalid bounding box (lies partially out of bounds)
        if tl[0] < 0 or tl[1] < 0 or \
           tl[0]+wh[0] >= W or tl[1]+wh[1] >= H:

                # If the box is completely out of bounds, return none.
                if tl[0] >= W or tl[1] >= H or tl[0]+wh[0] <= 0 or tl[1]+wh[1] <= 0:
                    return None

                # Access the valid pixels, then warp them into the destination buffer such that
                # they lie exactly where they would if the whole input box was valid.
                inner = max(0,int(tl[0])), max(0,int(tl[1])), min(W,int(tl[0]+wh[0])), min(H,int(tl[1]+wh[1]))
                inner_w,inner_h = inner[2] - inner[0], inner[3] - inner[1]
                buf = self.dset.ReadAsArray(inner[0],inner[1],inner_w,inner_h, buf_xsize=outw, buf_ysize=outh)
                #print(' - inner', inner, inner_w, inner_h, 'buf', buf)
                if buf is None: return None
                if nir:
                    assert buf.shape[0] == 4
                    buf = buf.transpose(1,2,0)[...,3:]
                elif buf.ndim == 3: buf = buf.transpose(1,2,0)[...,:3]
                in_pts = np.array((
                    inner[0],inner[1],
                    inner[2],inner[1],
                    inner[0],inner[3]),dtype=np.float32).reshape(3,2) * (outw/wh[0],outh/wh[1])
                out_pts = np.array((0,0, outw,0, 0,outh),dtype=np.float32).reshape(3,2)
                in_pts[:,0] -= tl[0] * (outw/wh[0])
                in_pts[:,1] -= tl[1] * (outh/wh[1])
                #M = cv2.getAffineTransform(in_pts.astype(np.float32),out_pts.astype(np.float32))
                M = cv2.getAffineTransform(out_pts.astype(np.float32),in_pts.astype(np.float32))
                return cv2.warpAffine(buf, M, (outw,outh))
        else:
            # Normal case where box lies entirely inside bounds
            buf = self.dset.ReadAsArray(int(tl[0]),int(tl[1]), int(wh[0]),int(wh[1]), buf_xsize=outw, buf_ysize=outh)

            # Automatically drop any channels >4
            if buf is None: return None
            if nir:
                assert buf.shape[0] == 4
                return buf.transpose(1,2,0)[...,3:]
            elif buf.ndim == 3: return buf.transpose(1,2,0)[...,:3]
            if buf.ndim == 2: return buf
            else:
                assert False, ('gdal read buffer had weird shape ' + buf.shape)

