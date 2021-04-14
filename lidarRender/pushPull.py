#from .gl_stuff import *
import cv2
import time
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
import sys

HW = 1024

def make_shader():
    vsrc = '''
    #version 440
    layout(location = 0) in vec3 i_pos;
    layout(location = 1) in vec2 i_uv;
    out vec2 v_uv;
    void main() {
        gl_Position = vec4(i_pos,1.0);
        v_uv = i_uv;
    }'''
    # Laplacian filter just to test with
    fsrc = '''
    #version 440
    in vec2 v_uv;
    layout(location = 0) out vec4 color;
    uniform sampler2D texSampler;
    void main() {
        vec3 c1 = texture(texSampler, v_uv).rgb;
        const float S = .02;
        vec3 c2 = texture(texSampler, v_uv + vec2(S,.0)).rgb;
        vec3 c3 = texture(texSampler, v_uv + vec2(-S,.0)).rgb;
        vec3 c4 = texture(texSampler, v_uv + vec2(.0,S)).rgb;
        vec3 c5 = texture(texSampler, v_uv + vec2(.0,-S)).rgb;
        c1 = c1 - (c2+c3+c4+c5) * .25;
        color = vec4(abs(c1), 1.0);
    }
    '''
    fsrc = '''
    #version 440
    in vec2 v_uv;
    layout(location = 0) out vec4 color;
    layout(binding = 0) uniform sampler2D texSampler;
    //layout(location = 1) uniform float T;
    layout(location = 2) uniform float S;
    void main() {
        vec2 uv = v_uv * S;
        vec3 c1 = texture(texSampler, uv).rgb;
        float t = S * T;
        vec3 c2 = texture(texSampler, uv + vec2(t,.0)).rgb;
        vec3 c3 = texture(texSampler, uv + vec2(-t,.0)).rgb;
        vec3 c4 = texture(texSampler, uv + vec2(.0,t)).rgb;
        vec3 c5 = texture(texSampler, uv + vec2(.0,-t)).rgb;
        //c1 = c1 - (c2+c3+c4+c5) * .25;
        //c1 = c1 - (c2+c3+c4+c5) * .075;
        c1 = c1;
        color = vec4(c1, 1.0);
    }
    '''
    vs = compileShader(vsrc, GL_VERTEX_SHADER)
    #fs = compileShader(fsrc, GL_FRAGMENT_SHADER)
    #return [compileProgram(vs,fs)]

    # Inefficient version: not making use of hardware bilinear sampling.
    h1 = np.array((
        1/64, 3/64,3/64, 1/64,
        3/64, 9/64,9/64, 3/64,
        3/64, 9/64,9/64, 3/64,
        1/64, 3/64,3/64, 1/64)).astype(np.float32).reshape(4,4) * 6
    h2 = np.array((
        1/16, 3/16,3/16, 1/16,
        3/16, 9/16,9/16, 3/16,
        3/16, 9/16,9/16, 3/16,
        1/16, 3/16,3/16, 1/16)).astype(np.float32).reshape(4,4) / 4
    print(' - h1 sum', h1.sum(), 'norm', np.linalg.norm(h1))
    print(' - h2 sum', h2.sum(), 'norm', np.linalg.norm(h2))

    def conv_down(v, out, h, step):
        code = ''
        code += ' vec4 a11 = h11 * texture(texSampler, uv + vec2(-STEP*2./2.,-STEP*2./2.));'
        code += ' vec4 a12 = h12 * texture(texSampler, uv + vec2(-STEP*2./2.,-STEP*1./2.));'
        code += ' vec4 a13 = h13 * texture(texSampler, uv + vec2(-STEP*2./2.,+STEP*1./2.));'
        code += ' vec4 a14 = h14 * texture(texSampler, uv + vec2(-STEP*2./2.,+STEP*2./2.));'
        code += ' vec4 a21 = h21 * texture(texSampler, uv + vec2(-STEP*1./2.,-STEP*2./2.));'
        code += ' vec4 a22 = h22 * texture(texSampler, uv + vec2(-STEP*1./2.,-STEP*1./2.));'
        code += ' vec4 a23 = h23 * texture(texSampler, uv + vec2(-STEP*1./2.,+STEP*1./2.));'
        code += ' vec4 a24 = h24 * texture(texSampler, uv + vec2(-STEP*1./2.,+STEP*2./2.));'
        code += ' vec4 a31 = h31 * texture(texSampler, uv + vec2(+STEP*1./2.,-STEP*2./2.));'
        code += ' vec4 a32 = h32 * texture(texSampler, uv + vec2(+STEP*1./2.,-STEP*1./2.));'
        code += ' vec4 a33 = h33 * texture(texSampler, uv + vec2(+STEP*1./2.,+STEP*1./2.));'
        code += ' vec4 a34 = h34 * texture(texSampler, uv + vec2(+STEP*1./2.,+STEP*2./2.));'
        code += ' vec4 a41 = h41 * texture(texSampler, uv + vec2(+STEP*2./2.,-STEP*3./2.));'
        code += ' vec4 a42 = h42 * texture(texSampler, uv + vec2(+STEP*2./2.,-STEP*1./2.));'
        code += ' vec4 a43 = h43 * texture(texSampler, uv + vec2(+STEP*2./2.,+STEP*1./2.));'
        code += ' vec4 a44 = h44 * texture(texSampler, uv + vec2(+STEP*2./2.,+STEP*2./2.));'
        code += ' vec4 b = a11 + a12 + a13 + a14 + a21 + a22 + a23 + a24  +'
        code += '          a31 + a32 + a33 + a34 + a41 + a42 + a43 + a44;'
        #code += ' float w = clamp(b.w,0.000000000001,999.);'
        code += ' float w = clamp(b.w,0.00001,999.);'
        code += ' float wc = 1. - pow((1. - clamp(w,0.,1.)), 16.0);'
        #code += ' float wc = clamp(w,0.,1.);'
        code += ' vec3 a = b.rgb / w;'
        code += ' if (lastLvl != 1) a = a * wc;'
        code += ' {} = vec4(a,wc);'.format(out)
        for i in range(1,5):
            for j in range(1,5): code = code.replace('h'+str(i)+str(j),str(h[i-1,j-1]))
        code = code.replace('STEP', step)
        return code
    def conv_up(v, out, h, step):
        code = ''
        code += ' vec4 a0 = texture(orig, 1*uv);'
        code += ' vec2 uv2 = uv*.5;'
        code += ' vec4 a11 = h11 * texture(texSampler, uv2 + vec2(-STEP*2./2.,-STEP*2./2.));'
        code += ' vec4 a12 = h12 * texture(texSampler, uv2 + vec2(-STEP*2./2.,-STEP*1./2.));'
        code += ' vec4 a13 = h13 * texture(texSampler, uv2 + vec2(-STEP*2./2.,+STEP*1./2.));'
        code += ' vec4 a14 = h14 * texture(texSampler, uv2 + vec2(-STEP*2./2.,+STEP*2./2.));'
        code += ' vec4 a21 = h21 * texture(texSampler, uv2 + vec2(-STEP*1./2.,-STEP*2./2.));'
        code += ' vec4 a22 = h22 * texture(texSampler, uv2 + vec2(-STEP*1./2.,-STEP*1./2.));'
        code += ' vec4 a23 = h23 * texture(texSampler, uv2 + vec2(-STEP*1./2.,+STEP*1./2.));'
        code += ' vec4 a24 = h24 * texture(texSampler, uv2 + vec2(-STEP*1./2.,+STEP*2./2.));'
        code += ' vec4 a31 = h31 * texture(texSampler, uv2 + vec2(+STEP*1./2.,-STEP*2./2.));'
        code += ' vec4 a32 = h32 * texture(texSampler, uv2 + vec2(+STEP*1./2.,-STEP*1./2.));'
        code += ' vec4 a33 = h33 * texture(texSampler, uv2 + vec2(+STEP*1./2.,+STEP*1./2.));'
        code += ' vec4 a34 = h34 * texture(texSampler, uv2 + vec2(+STEP*1./2.,+STEP*2./2.));'
        code += ' vec4 a41 = h41 * texture(texSampler, uv2 + vec2(+STEP*2./2.,-STEP*2./2.));'
        code += ' vec4 a42 = h42 * texture(texSampler, uv2 + vec2(+STEP*2./2.,-STEP*1./2.));'
        code += ' vec4 a43 = h43 * texture(texSampler, uv2 + vec2(+STEP*2./2.,+STEP*1./2.));'
        code += ' vec4 a44 = h44 * texture(texSampler, uv2 + vec2(+STEP*2./2.,+STEP*2./2.));'
        code += ' vec4 b = a11 + a12 + a13 + a14 + a21 + a22 + a23 + a24  +'
        code += '          a31 + a32 + a33 + a34 + a41 + a42 + a43 + a44;'
        #code += ' float wc = clamp(b.a/6.0 + a0.a, 0., 1.);' # WTF is this
        #code += ' float wc = a0.w;'
        code += ' float wc = a0.w;'
        code += ' vec3 a = (1.-wc) * b.rgb + a0.rgb;'
        #code += ' vec3 a = (1.-wc) * b.rgb;'
        #code += ' vec3 a =  texture(texSampler, uv).rgb;'
        #code += ' vec3 a =  a0.rgb;'
        #code += ' vec3 a = a0.rgb;'
        code += ' {} = vec4(a,1.);'.format(out)
        for i in range(1,5):
            for j in range(1,5): code = code.replace('h'+str(i)+str(j),str(h[i-1,j-1]))
        code = code.replace('STEP', step)
        return code


    # Push Shader
    push = '''
    #version 440
    in vec2 v_uv;
    layout(location = 0) out vec4 color;
    layout(binding = 0) uniform sampler2D texSampler;
    layout(location = 1) uniform float T;
    layout(location = 2) uniform float S;
    layout(location = 3) uniform float H;
    layout(location = 4) uniform int lastLvl;
    void main() {
        vec2 uv = S * v_uv;
        float step = S / (H);
    ''' + conv_down(1, 'color', h1, 'step') + '''
    }'''

    # Pull Shader
    pull = '''
    #version 440
    in vec2 v_uv;
    layout(location = 0) out vec4 color;
    layout(binding = 5) uniform sampler2D orig;
    layout(binding = 6) uniform sampler2D texSampler;
    layout(location = 2) uniform float S;
    layout(location = 3) uniform float H;
    void main() {
        vec2 uv = S * v_uv;
        float step = .5 * S / (H);
    ''' +conv_up(1, 'color', h2, 'step') + '''
    }
    '''

    # Copy-depth-to-alpha fragment shader
    copyDepth = '''
    #version 440
    in vec2 v_uv;
    layout(location = 0) out vec4 color;
    layout(binding = 3) uniform sampler2D renderedColor;
    layout(binding = 4) uniform sampler2D renderedDepth;
    layout(location = 2) uniform float S;
    void main() {
        vec2 uv = S * v_uv;
        vec3 c = texture(renderedColor, uv).rgb;
        float a = texture(renderedColor, uv).r;
        color = vec4(c,a);
    }
    '''

    simpleTextured = '''
    #version 440
    in vec2 v_uv;
    layout(location = 0) out vec4 color;
    layout(binding = 0) uniform sampler2D tex;
    void main() {
        vec2 uv = v_uv;
        color = texture(tex, uv);
    }
    '''

    push = push.replace('; ',';\n ')
    pull = pull.replace('; ',';\n ')
    copyDepth = copyDepth.replace('; ',';\n ')
    print(' PUSH SHADER\n',push)
    print(' PULL SHADER\n',pull)
    pull = compileShader(pull, GL_FRAGMENT_SHADER)
    push = compileShader(push, GL_FRAGMENT_SHADER)
    copyDepth = compileShader(copyDepth, GL_FRAGMENT_SHADER)
    simpleTextured = compileShader(simpleTextured, GL_FRAGMENT_SHADER)
    return [compileProgram(vs,push),compileProgram(vs,pull),compileProgram(vs,copyDepth),compileProgram(vs,simpleTextured)]



class PushPullGL():
    def __init__(self, w,h, maxWeight=1):
        self.w,self.h = w,h
        self.iw,ih = (1+3*w)//2,h
        glEnable(GL_BLEND)

        z,o = -1,1
        self.pts = np.array((z,z,0,0,0, z,o,0,0,1, o,o,0,1,1, o,o,0,1,1, o,z,0,1,0, z,z,0,0,0),dtype=np.float32)

        self.shaders = make_shader()

        '''
        self.texs = []
        while w > 8 and h > 8:
            tex = glGenTextures(1)
            self.texs.append(tex)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, w,h)
            w,h = w // 2, h // 2

            glTextureView
        '''

        m = glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS)
        print(' - MAX COLOR ATTACHMENTS:', m)

        # The texture that is the original render.
        self.tex0 = None

        # Ping-Pong textures
        texs = glGenTextures(10)
        self.tex0 = texs[0]
        self.texDepth = texs[1]
        self.texs = texs[2:]
        self.texOuts = glGenTextures(2)
        ww,hh = self.w,self.h
        for t in (*self.texs, *self.texOuts):
            glBindTexture(GL_TEXTURE_2D, t)
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, ww, hh)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT)
            #ww,hh = ww//2, hh//2
            #if ww < 2 or hh < 2: assert(False) # Screen must be greater than >512

        glBindTexture(GL_TEXTURE_2D, self.texDepth)
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT24, ww, hh)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.lvls = 8

        # Create framebuffers
        # fb0 is for the scene
        # fb1 outputs to tex1 OR tex2 with input from tex2 OR tex1
        self.fb0, self.fb1, self.fbOut = glGenFramebuffers(3)

        # NOTE XXX:
        # If we just render with alpha, then we can use the alpha as weight.
        # So I don't actually need to use the depth buffer (i.e. the shader copyDepth is not needed)
        # I'll still keep it though because it def could be useful
        glBindFramebuffer(GL_FRAMEBUFFER, self.fb0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.texs[0], 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, self.texDepth, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fb1)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.texs[0], 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, self.texs[1], 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, self.texs[2], 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, self.texs[3], 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, self.texs[4], 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, self.texs[5], 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, self.texs[6], 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT7, self.texs[7], 0)
        glDrawBuffers([GL_COLOR_ATTACHMENT0])
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbOut)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.texOuts[0], 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, self.texOuts[1], 0)
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.iters = 0

    def fullScreenQuad(self, S):
        glEnableVertexAttribArray(0); glEnableVertexAttribArray(1)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*5, self.pts)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*5, self.pts[3:])
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableVertexAttribArray(0); glDisableVertexAttribArray(1)


    def setRenderTarget(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fb0)
    def unsetRenderTarget(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def forwardWithRender(self):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texs[0])
        glBindFramebuffer(GL_FRAMEBUFFER, self.fb1)

        self.forward_(show=False, renderToScreen=True)

    # Here we UPLOAD CPU image to texs[0].
    def forwardWithImage(self, img, weight):
        x = np.concatenate((img,weight),-1).astype(np.float32)

        glBindTexture(GL_TEXTURE_2D, self.texs[0])
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, self.w,self.h, GL_RGBA, GL_FLOAT, x)

        glEnable(GL_TEXTURE_2D)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fb1)

        self.forward_(show=99999, renderToScreen=False)


    def forward_(self,show=False, renderToScreen=True):
        w,h = self.w,self.h

        # PUSH
        glUseProgram(self.shaders[0])
        #glUniform1f(1, .1*abs(np.sin(time.time()))**2)
        for i in range(1,self.lvls):
            w,h = w//2,h//2
            glUniform1f(2, 2./(1<<i))
            glUniform1f(3, float(h))
            glUniform1i(4, int(i == (self.lvls)-1))

            glBindTexture(GL_TEXTURE_2D, self.texs[i-1]) # Set input  texture
            glDrawBuffers([GL_COLOR_ATTACHMENT0+i])      # Set output texture

            glViewport(0,0,w,h)
            #print('pull',i,w,h)
            glClear(GL_COLOR_BUFFER_BIT)

            self.fullScreenQuad(i)
            #if i < 3: self.show('push_'+str(i),self.texs[i])

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbOut)

        # PULL
        glUseProgram(self.shaders[1])
        #glUniform1f(1, .1*abs(np.sin(time.time()))**2)
        for i in range(self.lvls-2,-1,-1):
            w,h = w*2,h*2
            #glUniform1f(2, (1<<(i+1)))
            glUniform1f(2, 1./(1<<(i)))
            glUniform1f(3, float(h/2))

            par = i % 2
            par1 = 1 - par

            glActiveTexture(GL_TEXTURE0)
            glDrawBuffers([GL_COLOR_ATTACHMENT0+par])           # Set output texture
            glActiveTexture(GL_TEXTURE6)
            if i == self.lvls-2: glBindTexture(GL_TEXTURE_2D, self.texs[i+1])
            #else: glBindTexture(GL_TEXTURE_2D, self.texs[i+1])
            else: glBindTexture(GL_TEXTURE_2D, self.texOuts[par1])
            glActiveTexture(GL_TEXTURE5)
            glBindTexture(GL_TEXTURE_2D, self.texs[i]) # Set input  texture
            #glBindTexture(GL_TEXTURE_2D, self.texOuts[1-(i%2)])


            glViewport(0,0,w,h)
            glClear(GL_COLOR_BUFFER_BIT)
            #print('push',i,w,h)
            #glClear(GL_COLOR_BUFFER_BIT)

            self.fullScreenQuad(i)
            if (show != False) and i < 3: self.show('pull_'+str(i),self.texOuts[par], time=-1)

        glActiveTexture(GL_TEXTURE5)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE6)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        self.iters += 1
        if show != False: cv2.waitKey(show)
        if show >1000: sys.exit(1)

        if renderToScreen:
            glUseProgram(self.shaders[3])
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glBindTexture(GL_TEXTURE_2D, self.texOuts[par])
            self.fullScreenQuad(0)

        glDisable(GL_TEXTURE_2D)



    def show(self, name, buf=None,time=1):
        if buf is None:
            #glBindTexture(GL_TEXTURE_2D, self.texs[2])
            glBindTexture(GL_TEXTURE_2D, self.texOuts[0])
        else:
            glBindTexture(GL_TEXTURE_2D, buf)

        y = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
        #y = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
        #y = np.frombuffer(y,dtype=np.uint8).reshape(self.h>>i, self.w>>i, -1)
        #y = np.frombuffer(y,dtype=np.uint8).reshape(self.h, self.w, -1)
        glBindTexture(GL_TEXTURE_2D, 0)
        cv2.imshow(name+'_y',y[...,:3])
        cv2.imshow(name+'_yweight',y[...,3])
        if time > 0: cv2.waitKey(time)


