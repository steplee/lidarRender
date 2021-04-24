import torch
import numpy as np

from OpenGL.GL import *
from OpenGL.GLUT import *

from .gl_stuff import *
from .data import get_dc_lidar

depth = 12

data = get_dc_lidar({'stride':2})
pts = data['pts']
inten = data['inten']
angle = data['angle']

from matplotlib.cm import inferno, bwr
N = pts.shape[0]
#colors = np.ones((N,4),dtype=np.float32)
colors = inferno(inten / inten.max()).astype(np.float32)
#colors = bwr(angle/40 + .5).astype(np.float32)
colors[...,3]=.71

app = OctreeApp((1000,1000))
app.init(True)
glEnable(GL_CULL_FACE)

for i in range(100000):
    app.updateCamera(.01)
    app.render()
    glColor4f(0,0,1,.5)

    glEnable(GL_BLEND)
    #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_DST_COLOR)
    #tree.render(10)

    draw_gizmo(2)

    glEnableClientState(GL_VERTEX_ARRAY)
    if pts is not None:
        #glColor4f(.6, .6, .99, .8)
        glColor4f(1, 1, 1, .7)

        glPointSize(2)
        glVertexPointer(3, GL_FLOAT, 0, pts)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4, GL_FLOAT, 0, colors)
        glDrawArrays(GL_POINTS, 0, len(pts))
        glDisableClientState(GL_COLOR_ARRAY)

    time.sleep(.008)
    glutSwapBuffers()
    glutPostRedisplay()
    glutMainLoopEvent()
    glFlush()

