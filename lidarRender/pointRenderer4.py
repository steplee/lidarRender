from .render import *
from .simple_dset import SimpleGeoDataset, gdal
import ctypes

from .occlusionAwareEngine import OcclusionAwareEngine

import pylas, os, numpy as np, torch, cv2, sys

DEBUG_MODE = True
DEBUG_MODE = False

def process_pc_get_pts(las, dset, stride=1):
    pts0 = np.stack((las.x[::stride], las.y[::stride], las.z[::stride]),-1).astype(np.float32)
    lo,hi = np.quantile(pts0, [.001,.999], 0)
    pts0 = pts0[(pts0[:,0] > lo[0]) & (pts0[:,1] > lo[1]) & (pts0[:,2] > lo[2]) & \
                (pts0[:,0] < hi[0]) & (pts0[:,1] < hi[1]) & (pts0[:,2] < hi[2]) ]
    #pts1 = ((pts0 - lo) / max(hi - lo)) * 2 - 1

    print(pts0.shape)
    print(' - Original Points')
    print(pts0)

    print(' - Tiff-Relative Points')
    dset_xform = dset.pix2native_
    dset_xform_inv = dset.native2pix_
    tl = dset_xform[:2,2]
    tw,th = dset.getRawSizeX(),dset.getRawSizeY()
    br = dset_xform @ np.array((tw,th,1.))
    xyxy = np.stack((tl,br),0)
    tlbr = np.concatenate((xyxy.min(0), xyxy.max(0)))

    pts2 = np.copy(pts0)
    pts2[...,:2] = pts0[...,:2] @ dset_xform_inv[:2,:2].T + dset_xform_inv[:2,2:].T
    pts2[...,2] = pts0[...,2] * dset_xform_inv[0,0]
    print(' - PTS 2 HERE 1 ',pts2.shape)
    pts2 = pts2[(pts2[:,0]>=0) & (pts2[:,1]>=0) & (pts2[:,0]<tw) & (pts2[:,1]<th)]
    print(' - PTS 2 HERE 1 ',pts2.shape)

    # Read tiff in pts AABB
    mins = pts2.min(0)
    maxs = pts2.max(0)
    print(' - Mins/Maxs:\n', mins,'\n',maxs)

    sx = 2048 * 2
    elevResMult = .25 * .75 * .5
    sy = ((maxs[1]-mins[1]) / (maxs[0]-mins[0])) * sx
    ss = sx / (maxs[0]-mins[0]) # The size we change to get output pix
    bb = (np.stack((mins[:2], (maxs-mins)[:2]),0)).reshape(-1)
    #bb[:2] -= (0,10)
    #bb[:2] += tlbr[:2]
    print(' - access bbox:', bb)
    #img = dset.bboxNative(bb, int(sx),int(sy))
    img = dset.bboxPix(bb, int(sy+1),int(sx))

    px = ((pts2[...,:2] - mins[:2]) / (.001+maxs-mins)[0]) * (sx,sx)
    print(' - Px:\n',px)
    print(' - Px max:',px.max(0))
    print(' - Px min:',px.min(0))
    print(' - img:',img.shape)
    px = px.astype(int)
    # TODO: Use torch grid_sample for bilinear sampling, or just implement with many numpy indexes
    colors = img[px[:,1],px[:,0]].astype(np.float32) / 255

    pts3 = ((pts2 - mins) / max(maxs-mins)) * 2 - 1

    #colors = (pts2 - mins) / (maxs-mins)


    print(' - Final pts:\n',pts3, pts3.dtype)
    print(' - Final colors:\n',colors,colors.dtype)

    pts3 = np.copy(pts3.astype(np.float32),'C')
    colors = np.copy(colors.astype(np.float32),'C')

    return pts3, colors, img


class PointRenderer(SingletonApp):
    def __init__(self, h, w):
        super().__init__((w,h), 'PointRenderer')
        self.w,self.h = w,h
        self.q_pressed = False
        self.velTrans = np.zeros(3, dtype=np.float32)
        self.accTrans = np.zeros(3, dtype=np.float32)

        #self.md = np.array((.0,.1))
        self.md = np.array((.0,.0))
        self.angles = np.array((0,0,0),dtype=np.float32)
        self.eye = np.array((-0,-0.0,1),dtype=np.float32)
        #self.eye = np.array((-0.5,-0.0,-.8),dtype=np.float32)

        self.R = np.eye(3,dtype=np.float32)
        self.t = np.copy(self.eye)
        self.view = np.eye(4,dtype=np.float32)

        self.pts = np.array([],dtype=np.float32)
        self.colors = np.array([],dtype=np.float32)
        self.vbo = None
        self.npts = 0
        self.vertSize = 0

        self.sky = np.random.randn(1000,3)
        self.sky = self.sky / np.linalg.norm(self.sky,axis=1)[:,np.newaxis]
        self.sky[...,2] = abs(self.sky[...,2])
        self.skyColors = np.clip(self.sky * (0,.2,.99), 0, 1)


    def do_init(self):
        meta = dict(w=self.w,h=self.h)
        self.engine = OcclusionAwareEngine(meta)

    def setPts(self, pts, colors=None, normals=None):
        pts = np.concatenate((pts, self.sky), 0)
        colors = np.concatenate((colors, self.skyColors), 0)

        self.haveColors = colors is not None
        self.haveNormals = normals is not None
        size = 3 + (3 if self.haveColors else 0) + (3 if self.haveNormals else 0)
        if self.vbo is None: self.vbo = glGenBuffers(1)
        arr = pts
        if self.haveColors: arr = np.hstack((arr,colors))
        if self.haveNormals: arr = np.hstack((arr,normals))
        arr = arr.astype(np.float32)
        print(arr.shape)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, size*arr.shape[0]*4, arr.reshape(-1), GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.npts = arr.shape[0]
        self.pts,self.colors,self.normals = pts,colors,normals
        self.vertSize = 6 * 4

    def renderPts(self, pts, colors):
        glUseProgram(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, self.vertSize, ctypes.c_void_p(0))
        glColorPointer(3, GL_FLOAT, self.vertSize, ctypes.c_void_p(4*3))
        glDrawArrays(GL_POINTS, 0, self.npts)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    def render(self):
        glViewport(0, 0, *self.wh)
        glMatrixMode(GL_PROJECTION)
        n = .02
        f = 4
        v = .5
        u = (v*self.wh[0]) / self.wh[1]
        glLoadIdentity()
        glFrustum(-u*n,u*n,-v*n,v*n,n,f)

        print(' - view:\n', self.view)
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(self.view.T.reshape(-1))

        glClearColor(0,0,0,0.)
        glClearDepth(1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        E = 1e-1


        glDepthRange(0,1.)
        if len(self.pts):
            glPointSize(1)

            #self.pp.setRenderTarget()
            #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            #self.renderPts(self.pts, self.colors)
            self.engine.setRenderTarget()

            #glBindFramebuffer(GL_FRAMEBUFFER,0)
            #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.renderPts(self.pts, self.colors)
            #self.drawTri()

            self.engine.render()
            self.engine.unsetRenderTarget()

    def drawTri(self):
        glUseProgram(0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(90*np.sin(time.time()), 0,0,1)
        glBegin(GL_TRIANGLES)
        z = -1.2
        glVertex3f(-.5,-.5,z)
        glVertex3f(.5,-.5,z)
        glVertex3f(.0,.75,z)
        glEnd()


    def startFrame(self):
        self.q_pressed = False
        glutMainLoopEvent()
        self.startTime = time.time()

        #self.md = self.md * .9 + .001 * (self.mprev - self.mprev2)
        self.md = self.md * .9 + .01 * np.array((self.left_dx, self.left_dy))
        self.velTrans = self.velTrans * .94 + .01 * self.accTrans
        self.angles = (self.md[1], self.md[0], 0)
        self.R = self.R @ cv2.Rodrigues(self.angles)[0].astype(np.float32)
        self.eye += self.R @ self.velTrans
        self.view[:3,:3] = self.R.T
        self.view[:3,3] = -self.R.T @ self.eye

        self.accTrans *= 0
        self.left_dx = self.left_dy = 0

    def endFrame(self):
        glutSwapBuffers()
        dt = time.time() - self.startTime
        st = .011 - dt
        #if st > 0: time.sleep(st)
        st = max(0,st)


        return self.q_pressed

    def keyboard(self, key, x, y):
        super().keyboard(key,x,y)
        key = (key).decode()
        if key == 'q': self.q_pressed = True
        if key == 'w': self.accTrans[2] = -1
        if key == 's': self.accTrans[2] = 1

    def motion(self, x, y):
        super().motion(x,y)



def main():
    dset = SimpleGeoDataset('/data/pointCloudStuff/img/DC/2013.utm.tif')

    dir_ = '/data/pointCloudStuff'
    ext = '.las'

    app = PointRenderer(1024,1024)
    app.init(True)

    for fname in [f for f in os.listdir(dir_) if f.endswith(ext)]:
        fname = os.path.join(dir_,fname)
        print(' - Opening', fname)
        with pylas.open(fname) as fp:
            las = fp.read()

            stride = 1
            '''
            pts0 = np.stack((las.x[::stride], las.y[::stride], las.z[::stride]),-1).astype(np.float32)
            lo,hi = np.quantile(pts0, [.003,.997], 0)
            pts0 = pts0[(pts0[:,0] > lo[0]) & (pts0[:,1] > lo[1]) & (pts0[:,2] > lo[2]) & \
                        (pts0[:,0] < hi[0]) & (pts0[:,1] < hi[1]) & (pts0[:,2] < hi[2]) ]
            pts1 = ((pts0 - lo) / max(hi - lo)) * 2 - 1
            #pts = torch.from_numpy(pts1).cuda()

            app.pts = pts1
            app.colors = (pts0 - lo) / (hi - lo)
            '''
            pts, colors, img = process_pc_get_pts(las, dset, stride)
            #app.pts,app.colors = pts,colors
            app.setPts(pts,colors,None)

            print(' - Setting', app.pts.shape[0],'pts')

            while not app.q_pressed:
                app.startFrame()
                app.render()
                if app.endFrame(): break
            app.q_pressed = False


if __name__ == '__main__':
    main()



