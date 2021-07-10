from .render import *
from .simple_dset import SimpleGeoDataset, gdal
import ctypes

from .occlusionAwareEngine import OcclusionAwareEngine

import pylas, os, numpy as np, torch, cv2, sys
from matplotlib.cm import inferno, autumn

import torch
import torch.utils.cpp_extension
print(' - Compiling cuda/c extension')
voxelize = torch.utils.cpp_extension.load(
        name='voxelize',
        sources=[
            'heightmapMeshing/csrc/voxelize.cu',
            'heightmapMeshing/csrc/surface_net.cc',
            'heightmapMeshing/csrc/surface_net.cu',
            'heightmapMeshing/csrc/normals.cu',
            'heightmapMeshing/csrc/utils.cu'
            ],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=0', '-g'],
        extra_cuda_cflags=['-D_GLIBCXX_USE_CXX11_ABI=0'],
        extra_ldflags=['-lcusolver'],
        verbose=True
        #verbose=False
        )

DEBUG_MODE = True
DEBUG_MODE = False

def process_pc_get_pts(las, dset, stride=1):
    pts0 = np.stack((las.x[::stride], las.y[::stride], las.z[::stride]),-1).astype(np.float32)
    lo,hi = np.quantile(pts0, [.2,.8], 0)
    #lo,hi = np.quantile(pts0, [.0015,.9985], 0)
    loz,hiz = np.quantile(pts0, [.0015,.9985], 0)
    pts0 = pts0[(pts0[:,0] > lo[0]) & (pts0[:,1] > lo[1]) & (pts0[:,2] > loz[2]) & \
                (pts0[:,0] < hi[0]) & (pts0[:,1] < hi[1]) & (pts0[:,2] < hiz[2]) ]
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
        self.eye = np.array((-0,-0.0,0),dtype=np.float32)
        #self.eye = np.array((.5,.50,.5),dtype=np.float32)
        #self.eye = np.array((-0.5,-0.0,-.8),dtype=np.float32)

        self.R = np.eye(3,dtype=np.float32)
        self.t = np.copy(self.eye)
        self.view = np.eye(4,dtype=np.float32)

        self.pts = np.array([],dtype=np.float32)
        self.colors = np.array([],dtype=np.float32)
        self.vbo = None
        self.npts = 0
        self.vertSize = 0
        self.renderIters = 1
        self.vbo2 = None
        self.haveNormals = False

        #self.sky = np.random.randn(1000,3)
        #self.sky = self.sky / np.linalg.norm(self.sky,axis=1)[:,np.newaxis]
        #self.sky[...,2] = abs(self.sky[...,2])
        #self.skyColors = np.clip(self.sky * (0,.2,.99), 0, 1)


    def do_init(self):
        meta = dict(w=self.w,h=self.h)
        self.engine = OcclusionAwareEngine(meta)
        self.DO_PROCESS = True
        #self.DO_PROCESS = False
        #self.ONLY_DEPTH_FILTER = True
        self.ONLY_DEPTH_FILTER = False
        self.haveTris = False

    def setTris(self, verts, colors, tris, normalPts=None,normals=None):
        self.haveTris = True
        if self.vbo is not None: glDeleteBuffers(1,[self.vbo])
        self.vbo,self.vbo2 = glGenBuffers(2)
        arr = np.hstack((verts,colors))
        self.ibo,self.ibo2 = glGenBuffers(2)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, arr.shape[1]*arr.shape[0]*4, arr.reshape(-1), GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, tris.size*4, tris.reshape(-1), GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        self.vertSize = arr.shape[1]*4
        self.ninds = tris.size

        arr = verts
        inds = np.hstack((
            tris[::3][:,np.newaxis], tris[1::3][:,np.newaxis],
            tris[1::3][:,np.newaxis], tris[2::3][:,np.newaxis],
            tris[::3][:,np.newaxis], tris[2::3][:,np.newaxis]))
        print('INDS',inds.shape)
        self.nindsLines = inds.size
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo2)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, inds.size*4, inds.reshape(-1), GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        if normals is not None:
            arr = np.hstack((normalPts, normalPts+normals*.003))
            print('normalPts:\n',normalPts)
            print('normals:\n',normals)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo2)
            glBufferData(GL_ARRAY_BUFFER, arr.shape[1]*arr.shape[0]*4, arr.reshape(-1), GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.nnormals = arr.shape[0] * 2
            self.haveNormals = True

    def renderTris(self):
        self.drawAxes()
        glEnable(GL_CULL_FACE)
        if self.renderIters % 200 == 0: print(' - Rendering', self.ninds, 'inds')
        glUseProgram(0)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1,1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, self.vertSize, ctypes.c_void_p(0))
        glColorPointer(3, GL_FLOAT, self.vertSize, ctypes.c_void_p(12))
        glDrawElements(GL_TRIANGLES, self.ninds, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glDisableClientState(GL_COLOR_ARRAY)
        glDisable(GL_POLYGON_OFFSET_FILL)

        glColor4f(1,1,1,.3)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo2)
        glVertexPointer(3, GL_FLOAT, self.vertSize, ctypes.c_void_p(0))
        glDrawElements(GL_LINES, self.nindsLines, GL_UNSIGNED_INT, ctypes.c_void_p(0))

        if self.haveNormals:
            glColor4f(.7,.3,1,.3)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo2)
            glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
            glDrawArrays(GL_LINES, 0, self.nnormals)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            print(' - rendering', self.nnormals, 'normals')
        glColor4f(1,1,1,1)


        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def setPts(self, pts, colors=None, normals=None):
        #pts = np.concatenate((pts, self.sky), 0)
        #colors = np.concatenate((colors, self.skyColors), 0)

        self.haveColors = colors is not None
        self.haveNormals = normals is not None
        size = 3 + (3 if self.haveColors else 0) + (3 if self.haveNormals else 0)
        if self.vbo is not None: glDeleteBuffers(1,[self.vbo])
        self.vbo = glGenBuffers(1)
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
        if self.haveColors:
            self.vertSize = 6 * 4
        else:
            self.vertSize = 3 * 4

        self.lo = pts.min(0)
        self.hi = pts.max(0)
        self.sz = (self.hi-self.lo)[:2].max()

    def renderPts(self, pts, colors):
        #self.drawAxes()
        glUseProgram(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, self.vertSize, ctypes.c_void_p(0))
        if self.haveColors:
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, self.vertSize, ctypes.c_void_p(4*3))
        glDrawArrays(GL_POINTS, 0, self.npts)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    def render(self):
        glViewport(0, 0, *self.wh)
        glMatrixMode(GL_PROJECTION)
        if self.DO_PROCESS or self.ONLY_DEPTH_FILTER:
            glLoadIdentity()
            lo,hi,sz = self.lo,self.hi,self.sz
            print(lo,hi)
            n,f = abs(hi[2]), abs(lo[2])
            n = .8
            print(' - NEAR FAR', n, f)
            f=1
            #n,f = 1e-4, abs(lo[2])
            #n,f = 1, 1.5
            glOrtho(lo[0],hi[0], lo[1],hi[1], n,f)
        else:
            n = .001
            f = 4
            v = .5
            u = (v*self.wh[0]) / self.wh[1]
            glLoadIdentity()
            glFrustum(-u*n,u*n,-v*n,v*n,n,f)

        if self.renderIters % 60 == 0:
            print(' - view:\n', self.view)
        self.renderIters += 1
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
            if self.DO_PROCESS or self.ONLY_DEPTH_FILTER: self.engine.setRenderTarget()

            #glBindFramebuffer(GL_FRAMEBUFFER,0)
            #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            if self.haveTris:
                self.renderTris()
            else:
                self.renderPts(self.pts, self.colors)
            #self.drawTri()

            if self.ONLY_DEPTH_FILTER:
                self.engine.render(True)
                self.engine.unsetRenderTarget()

            elif self.DO_PROCESS:
                self.engine.render(False)

                glViewport(0,0,self.w,self.h)
                data = glReadPixels(0,0,self.w,self.h, GL_RGBA, GL_FLOAT)
                data = torch.from_numpy(data).cuda().float()
                elev = data[...,0] \
                    .mul_(-1).add_(1) \
                    #.sub_(n).div_(f)
                data[0:4].fill_(0)
                data[-4:].fill_(0)
                data[:,:4].fill_(0)
                data[:,-4:].fill_(0)

                self.engine.unsetRenderTarget()

                #elev.mul_(.9)
                #elev[elev>.3] = .1
                elev = elev.contiguous()
                vv = voxelize.forward(elev, 1000)
                print(elev.shape)
                print(vv.shape)
                print('VV\n',vv,'VV')
                #vv_ = vv.cpu().numpy().astype(np.uint8)
                if 0:
                    elev_ = elev.cpu().numpy()
                    cv2.imshow('ELEV', elev_)
                    #cv2.imshow('VV', vv_)
                    cv2.waitKey(1)
                    newPts = vv[:,:3].float() / data.shape[0]
                    newPts = newPts.cpu().numpy()
                    #newColors = np.ones((newPts.shape[0],4),dtype=np.float32)
                    newColors = np.ones((newPts.shape[0],3),dtype=np.float32)
                    newColors[:,:3] = inferno(.1 + newPts[:,2:3] / newPts[:,2:3].max())[...,0,:3]
                    newPts = newPts * 2 - 1
                    print(' newPts max', newPts.max(0))
                    print(' newPts min', newPts.min(0))
                    self.setPts(newPts,newColors)
                else:
                    print(' - Making geo')
                    #verts,tris,quads = voxelize.makeGeo(vv, data.shape[0])
                    verts_t,tris,quads = voxelize.makeGeoSurfaceNet(vv, data.shape[0])
                    verts = verts_t.cpu().numpy().reshape(-1,3)
                    colors = np.ones_like(verts)
                    colors[:,:3] = inferno(verts[:,2:3] / verts[:,2:3].max())[...,0,:3]
                    tris = tris.cpu().numpy()
                    print(' - Tris:\n', tris)
                    print(verts.shape,colors.shape,tris.shape, tris.max())
                    print('max vert', verts.max(0))
                    print('min vert', verts.min(0))

                    normalPts,normals = voxelize.estimateSurfaceTangents(verts_t.cuda())
                    normalPts = normalPts.cpu().numpy()
                    normals = normals.cpu().numpy()

                    self.setTris(verts,colors,tris, normalPts,normals)

                self.DO_PROCESS = False
                cv2.waitKey(100)


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
    def drawAxes(self):
        glUseProgram(0)
        glBegin(GL_LINES)
        s = 1
        zoff = 0
        glColor4f(1,0,0,.65); glVertex3f(0,0,zoff); glVertex3f(s,0,zoff);
        glColor4f(0,1,0,.65); glVertex3f(0,0,zoff); glVertex3f(0,s,zoff);
        glColor4f(0,0,1,.65); glVertex3f(0,0,zoff); glVertex3f(0,0,s+zoff);
        glEnd()
        glColor4f(1,1,1,1)



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
        #cv2.waitKey(1)


        return self.q_pressed

    def keyboard(self, key, x, y):
        super().keyboard(key,x,y)
        key = (key).decode()
        if key == 'q': self.q_pressed = True
        if key == 'w': self.accTrans[2] = -1
        if key == 's': self.accTrans[2] = 1
        if key == 'j': self.accTrans[1] = -1
        if key == 'k': self.accTrans[1] = 1
        if key == 'a': self.accTrans[0] = -1
        if key == 'd': self.accTrans[0] = 1
        self.accTrans /= 8

    def motion(self, x, y):
        super().motion(x,y)



def main_lidar():
    #dset = SimpleGeoDataset('/data/pointCloudStuff/img/DC/2013.utm.tif')
    dset = SimpleGeoDataset('/data/dc_tiffs/dc.tif')

    dir_ = '/data/pointCloudStuff/pc/dc'
    ext = '.las'

    #app = PointRenderer(1024//2,1024//2)
    app = PointRenderer(1024,1024)
    #app = PointRenderer(1024*2,2*1024)

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
            pts,density = voxelize.filterOutliers(torch.from_numpy(pts), 12)
            density = density.cpu().numpy()
            pts = pts.cpu().numpy()
            T = 2
            pts = pts[density>T]
            colors = autumn(density[density>T] / 5)[...,:3].astype(np.float32)
            #colors[density<1.7] = (0,1,0)

            #app.pts,app.colors = pts,colors
            app.setPts(pts,colors,None)

            print(' - Setting', app.pts.shape[0],'pts')

            while not app.q_pressed:
                app.startFrame()
                app.render()
                app.endFrame()
                #if app.endFrame(): break
            app.q_pressed = False

def main_synth():
    n = 1536
    n = 2000
    pts = np.stack(np.meshgrid(
        np.linspace(-1,1,n),
        np.linspace(-1,1,n),
        [-1]), -1).reshape(-1,3)

    pts = pts.astype(np.float32)
    pts[...,2] = np.random.randn(n*n) * .0001 - .9

    pts[
        (pts[:,1]+pts[:,0] > -.5) &
        (pts[:,1]+pts[:,0] <  .5) &
        (pts[:,1]-2*pts[:,0] > -.5) &
        (2*pts[:,1]-pts[:,0] <  .5) ,2 ] = -.6
    pts[
        (pts[:,1]+pts[:,0] > -.3) &
        (pts[:,1]+pts[:,0] <  .3) &
        (pts[:,1]-pts[:,0] > -.3) &
        (pts[:,1]-pts[:,0] <  .3) ,2 ] = -.4
    print('PTS\n',pts)

    lo = pts[:,2].min()
    hi = pts[:,2].max()
    print(lo,hi)
    colors = inferno(1e-1 + (pts[:,2] - lo) / (1e-6+hi - lo))[...,:3]

    app = PointRenderer(1024,1024)
    app.init(True)
    app.setPts(pts,colors,None)
    while not app.q_pressed:
        app.startFrame()
        app.render()
        app.endFrame()
        #if app.endFrame(): break
    app.q_pressed = False

if __name__ == '__main__':
    main_lidar()
    #main_synth()




