import cv2
import time
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
import sys


def makeShaders(nlvls):
    vsrc = '''
    #version 440
    layout(location = 0) in vec3 i_pos;
    layout(location = 1) in vec2 i_uv;
    out vec2 v_uv;
    void main() {
        gl_Position = vec4(i_pos,1.0);
        v_uv = i_uv;
    }'''
    vs = compileShader(vsrc, GL_VERTEX_SHADER)
    unproject = '''
    #version 440
    in vec2 v_uv;
    out vec4 xyz;
    layout(location = 4) uniform mat4 invProjT;
    layout(binding = 0) uniform sampler2D depthMap;

    const float twoThirds = 2./3.;
    const float threeHalfs = 3./2.;

    void main() {
        //vec2 uv = (3./2.) * v_uv;
        vec2 uv = v_uv;
        //float x = uv.x * 2. - 1.;
        //float y = uv.y * 2. - 1.;
        float x = uv.x;
        float y = uv.y;

        //x=y=0; // NOTE.

        //float z = texture(depthMap, (2./3.) * v_uv).r * 2 - 1.;
        float z = texture(depthMap, (2./3.) * v_uv).r;
        xyz = (invProjT *  vec4(x,y, z, 1));

        /*
            //float z = texture(depthMap, v_uv).r;
            //float z = (texture(depthMap, v_uv).r + 1.) * .5;
            float z0 = texture(depthMap, v_uv).r;
            //float z = (z0 * 2.) - 1.;
            float z = z0;

            //xyz = (invProj * vec4(x,y, 1., -z));
            //xyz = (invProj * z * vec4(x,y, 1., -1.));

            //xyz = (invProj *  vec4(x,y, z, 1));

            float far = 4.;
            float near = .02;
            float zz = (2*far*near/(far-near)) / (z0 * -2. + 1. + (far+near) / (far-near));
            zz = (zz - near) / (far - near);
            xyz = vec4(x*invProj[0][0], y*invProj[1][1], -zz, 1.);
        */

        xyz.xyz = xyz.xyz / xyz.w;

        // NOTE: Dividing by far plane here.
        //xyz.z *= -1 / 4.2;

        xyz.z *= -1; // NOTE: Assumes far plane is 1
        //xyz.z = 1. - (2. / (1.+xyz.z) - 1.);
        //xyz.z *= 1. - .88;

        xyz.a = 1.0;
    }
    '''
    pyrDownDepth = '''
    #version 440
    in vec2 v_uv;
    layout(location = 0) out vec4 xyz;
    layout(binding = 0) uniform sampler2D texXYZ;
    //layout(location = 2) uniform float S;
    //layout(location = 3) uniform float H;

    layout(location = 2) uniform float W;
    layout(location = 3) uniform int lvl0;

    const float twoThirds = 2./3.;
    const float threeHalfs = 3./2.;

    void main() {

        int lvl1 = lvl0 - 1;

        //float P = (1. / (1<<(lvl1+1)));
        float P = (1. / (1<<(lvl1+1)));
        float uoff = .5 * (1 - pow(.25,(lvl1+1)/2)) / (1.-.25);
        float voff = .25 * (1 - pow(.25,(lvl1)/2)) / (1.-.25);

        vec2 uv1 = (4./3.) * (P * v_uv + vec2(uoff, voff));

        // I think the step is constant. Although the access in terms of level=0
        // is (1<<lvl)/W, the access of the above level is fixed per lvl?
        //float step = (1<<lvl1) / W;
        float step = (2./3.) / W;

        vec4 p11 = texture(texXYZ, uv1 + vec2(0.,0.));
        vec4 p12 = texture(texXYZ, uv1 + vec2(0.,step));
        vec4 p21 = texture(texXYZ, uv1 + vec2(step,0.));
        vec4 p22 = texture(texXYZ, uv1 + vec2(step,step));

        if (p11.z < p12.z && p11.z < p21.z && p11.z < p22.z) xyz = p11;
        else if (p12.z < p21.z && p12.z < p22.z) xyz = p12;
        else if (p21.z < p22.z) xyz = p21;
        else xyz = p22;
        //if (p11.a > .6 && p11.z < p12.z && p11.z < p21.z && p11.z < p22.z) xyz = p11;
        //else if (p12.a > .6 && p12.z < p21.z && p12.z < p22.z) xyz = p12;
        //else if (p21.a > .6 && p21.z < p22.z) xyz = p21;
        //else xyz = p22;

        //xyz.a = 1.;
        //xyz = vec4(lvl0/8.,lvl0/8.,lvl0/8.,1.);
        //xyz = vec4(uv1.s,uv1.t,1.,1.).bgra;
        //xyz = vec4(v_uv.s,v_uv.t,1.,1.).bgra;

    }'''
    getMaskOld = '''
    #version 440
    in vec2 v_uv;
    layout(location = 0) out vec4 outColor;
    layout(binding = 0) uniform sampler2D xyzEven;
    layout(binding = 1) uniform sampler2D xyzOdd;
    layout(binding = 2) uniform sampler2D inColor;
    //layout(location = 2) uniform float S;
    layout(location = 3) uniform float W;

    void main() {

        // Read depth map to est scale.
        //int lvl0 = 1;
        int lvl0 = 0;
        float P0 = (1. / (1<<(lvl0+1)));
        float uoff0 = .5 * (1 - pow(.25,(lvl0+1)/2)) / (1.-.25);
        float voff0 = .25 * (1 - pow(.25,(lvl0)/2)) / (1.-.25);
        vec2 uv0 = (4./3.) * (P0 * v_uv + vec2(uoff0,voff0));

        float s_hpr = 4;
        float h_div_2tant = 1;
        //float z = (1. - texture(xyzLvl4, uv0).z / 4.2);
        //float z = texture(xyzOdd, uv0).z;
        float z = texture(xyzEven, uv0).z;

        //float lvl_ = (1-z) * (NLVL + .2);
        //float lvl_ = 1. / (7.1*z);
        float lvl_ = 1 * log(s_hpr * h_div_2tant * (1./z)) / log(2.);

        lvl_ = clamp(lvl_, 0., (NLVL + .01));
        int lvl = int(lvl_);

        vec3 x;
        x = texture(xyzEven, (2./3.) * v_uv).xyz;
        float meanOcc = 0.;

            // Single Level Version
            /*
            float P1 = (1. / (1<<(lvl+1)));
            float uoff1 = .5 * (1 - pow(.25,(lvl+1)/2)) / (1.-.25);
            float voff1 = .25 * (1 - pow(.25,(lvl)/2)) / (1.-.25);
            vec2 uv1 = (4./3.) * (P1 * v_uv + vec2(uoff1,voff1));
            float step = (2./3.) / W;


            for (int dy=-1; dy<2; dy++)
            for (int dx=-1; dx<2; dx++) {
                if (dx != 0 || dy != 0) {
                    //vec2 uv2 = uv1 + vec2(dx * step + off, dy * step + off);
                    vec2 uv2 = uv1 + vec2(dx * step, dy * step);

                    vec3 y;
                    if (lvl % 2 == 0) y = texture(xyzEven, uv2).xyz;
                    else              y = texture(xyzOdd, uv2).xyz;

                    float occ = 1. - dot(normalize(y-x), normalize(-y));
                    meanOcc += occ * (1./8.);
                }
            }
            */

            float sectors[8] = float[](999,999,999,999,999,999,999,999);
            for (int lvl1=0; lvl1<lvl+1; lvl1++) {
                float P1 = (1. / (1<<(lvl1+1)));
                float uoff1 = .5 * (1 - pow(.25,(lvl1+1)/2)) / (1.-.25);
                float voff1 = .25 * (1 - pow(.25,(lvl1)/2)) / (1.-.25);
                vec2 uv1 = (4./3.) * (P1 * v_uv + vec2(uoff1,voff1));
                float step = (2./3.) / W;
                int ii = 0;

                for (int dy=-1; dy<2; dy++)
                for (int dx=-1; dx<2; dx++) {
                    if (dx != 0 || dy != 0) {
                        //vec2 uv2 = uv1 + vec2(dx * step + off, dy * step + off);
                        vec2 uv2 = uv1 + vec2(dx * step, dy * step);

                        vec3 y;
                        if (lvl1 % 2 == 0) y = texture(xyzEven, uv2).xyz;
                        else              y = texture(xyzOdd, uv2).xyz;

                        float occ = 1. - dot(normalize(y-x), normalize(-y));
                        if (occ < sectors[ii]) sectors[ii] = occ;
                        ii += 1;
                    }
                }
            }
            for (int i=0; i<8; i++) meanOcc += (1./8.) * sectors[i];





        vec4 c0 = vec4(texture(xyzEven, (2./3.) * v_uv).zzz, 1.0);

        if (c0.z <= 0.0000001 || c0.z >= .9999999) c0.a = 0;
        if (meanOcc > .5) c0 = vec4(0,1,0,1);
        //if (c0.z >= .9999999) c0 = vec4(1.,1,0,1.);
        //if (c0.z <= 0.0000001) c0 = vec4(0.,0,1,1.);


        //vec4 c0 = texture(inColor, (2./3.) * v_uv);
        //vec4 c0 = texture(inColor, (2./3.) * v_uv);
        outColor = c0;
    }
    '''.replace('NLVL', str(nlvls))
    getMask= '''
    #version 440
    in vec2 v_uv;
    layout(location = 0) out vec4 outColor;
    layout(binding = 0) uniform sampler2D xyzEven;
    layout(binding = 1) uniform sampler2D xyzOdd;
    layout(binding = 2) uniform sampler2D inColor;
    //layout(location = 2) uniform float S;
    layout(location = 3) uniform float W;

    void main() {

        // For the four directions, walk 5 pixels over (20 total)
        // We want for at least one direction NOT to hit an obstruction.

        float myZ = texture(xyzEven, v_uv * (2./3.)).z;
        //float T = .00001;
        float T = .001;
        float S = 1;
        int M = 8;
        float sign = -1;

        float meanOcc = 4;
        for (int d=1; d<M; d++) {
            float z = texture(xyzEven, (2./3.) * (v_uv + vec2(S*float(d)/W, 0))).z;
            if (z > .99999) continue;
            if (sign*(z-myZ) > (d) * T) {
                meanOcc -= 1;
                break;
            }
        }
        for (int d=1; d<M; d++) {
            float z = texture(xyzEven, (2./3.) * (v_uv + vec2(S*float(-d)/W, 0))).z;
            if (z > .99999) continue;
            if (sign*(z-myZ) > (d) * T) {
                meanOcc -= 1;
                break;
            }
        }
        for (int d=1; d<M; d++) {
            float z = texture(xyzEven, (2./3.) * (v_uv + vec2(0, S*float(d)/W))).z;
            if (z > .99999) continue;
            if (sign*(z-myZ) > (d) * T) {
                meanOcc -= 1;
                break;
            }
        }
        for (int d=1; d<M; d++) {
            float z = texture(xyzEven, (2./3.) * (v_uv + vec2(0, S*float(-d)/W))).z;
            if (z > .99999) continue;
            if (z < .00001) continue;
            if (sign*(z-myZ) > (d) * T) {
                meanOcc -= 1;
                break;
            }
        }


        if (myZ > .9999999) meanOcc = 4;


        vec4 c0 = vec4(texture(xyzEven, (2./3.) * v_uv).zzz, 1.0);

        if (c0.z <= 0.0000001 || c0.z >= .9999999) c0.a = 0;

        //c0 = vec4(meanOcc,meanOcc,meanOcc,1.);
        //if (myZ == 1) c0.a = 0;
        //if (myZ != 0 && myZ != 1 && meanOcc <1.1) c0 = vec4(0,1,0,1);
        if (myZ != 1 && meanOcc <1.1) c0 = vec4(0,1,0,0);
        if (myZ == 0 || myZ == 1) c0.a = 0;

        //if (c0.z >= .9999999) c0 = vec4(1.,1,0,1.);
        //if (c0.z <= 0.0000001) c0 = vec4(0.,0,1,1.);


        //vec4 c0 = texture(inColor, (2./3.) * v_uv);
        //vec4 c0 = texture(inColor, (2./3.) * v_uv);
        outColor = c0;
    }
    '''.replace('NLVL', str(nlvls))

    simpleTextured = '''
    #version 440
    in vec2 v_uv;
    layout(location = 0) out vec4 color;
    layout(binding = 0) uniform sampler2D tex;
    void main() {
        vec2 uv = (1) * v_uv;
        color = texture(tex, uv);
    }
    '''

    unproject = unproject.replace('; ',';\n ')
    pyrDownDepth = pyrDownDepth.replace('; ',';\n ')
    getMask = getMask.replace('; ',';\n ')
    #print(' unproject SHADER\n',unproject)
    #print(' pyrDownDepth SHADER\n',pyrDownDepth)
    #print(' getMask SHADER\n',getMask)
    unproject = compileShader(unproject, GL_FRAGMENT_SHADER)
    pyrDownDepth = compileShader(pyrDownDepth, GL_FRAGMENT_SHADER)
    getMask = compileShader(getMask, GL_FRAGMENT_SHADER)
    simpleTextured = compileShader(simpleTextured, GL_FRAGMENT_SHADER)



    unproject = compileProgram(vs,unproject)
    pyrDownDepth = compileProgram(vs,pyrDownDepth)
    getMask = compileProgram(vs,getMask)
    simpleTextured = compileProgram(vs,simpleTextured)
    return locals()

def makeShaders_pull_push(vs, nlvls):
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
        code += ' vec4 a11 = h11 * texture(lastImg, uv + vec2(-STEP*2./2.,-STEP*2./2.));'
        code += ' vec4 a12 = h12 * texture(lastImg, uv + vec2(-STEP*2./2.,-STEP*1./2.));'
        code += ' vec4 a13 = h13 * texture(lastImg, uv + vec2(-STEP*2./2.,+STEP*1./2.));'
        code += ' vec4 a14 = h14 * texture(lastImg, uv + vec2(-STEP*2./2.,+STEP*2./2.));'
        code += ' vec4 a21 = h21 * texture(lastImg, uv + vec2(-STEP*1./2.,-STEP*2./2.));'
        code += ' vec4 a22 = h22 * texture(lastImg, uv + vec2(-STEP*1./2.,-STEP*1./2.));'
        code += ' vec4 a23 = h23 * texture(lastImg, uv + vec2(-STEP*1./2.,+STEP*1./2.));'
        code += ' vec4 a24 = h24 * texture(lastImg, uv + vec2(-STEP*1./2.,+STEP*2./2.));'
        code += ' vec4 a31 = h31 * texture(lastImg, uv + vec2(+STEP*1./2.,-STEP*2./2.));'
        code += ' vec4 a32 = h32 * texture(lastImg, uv + vec2(+STEP*1./2.,-STEP*1./2.));'
        code += ' vec4 a33 = h33 * texture(lastImg, uv + vec2(+STEP*1./2.,+STEP*1./2.));'
        code += ' vec4 a34 = h34 * texture(lastImg, uv + vec2(+STEP*1./2.,+STEP*2./2.));'
        code += ' vec4 a41 = h41 * texture(lastImg, uv + vec2(+STEP*2./2.,-STEP*3./2.));'
        code += ' vec4 a42 = h42 * texture(lastImg, uv + vec2(+STEP*2./2.,-STEP*1./2.));'
        code += ' vec4 a43 = h43 * texture(lastImg, uv + vec2(+STEP*2./2.,+STEP*1./2.));'
        code += ' vec4 a44 = h44 * texture(lastImg, uv + vec2(+STEP*2./2.,+STEP*2./2.));'
        code += ' vec4 b = a11 + a12 + a13 + a14 + a21 + a22 + a23 + a24  +'
        code += '          a31 + a32 + a33 + a34 + a41 + a42 + a43 + a44;'
        #code += ' float w = clamp(b.w,0.000000000001,999.);'
        code += ' float w = clamp(b.w,0.00001,999.);'
        code += ' float wc = 1. - pow((1. - clamp(w,0.,1.)), 16.0);'
        #code += ' float wc = clamp(w,0.,1.);'
        code += ' vec3 a = b.rgb / w;'
        code += ' if (isLastLvl != 1) a = a * wc;'
        code += ' {} = vec4(a,wc);'.format(out)
        for i in range(1,5):
            for j in range(1,5): code = code.replace('h'+str(i)+str(j),str(h[i-1,j-1]))
        code = code.replace('STEP', step)
        return code
    def conv_up(v, out, h, step):
        code = ''
        code += ' vec4 a0 = texture(img, uv);'
        code += ' vec4 a11 = h11 * texture(nxtImg, uv2 + vec2(-STEP*2./2.,-STEP*2./2.));'
        code += ' vec4 a12 = h12 * texture(nxtImg, uv2 + vec2(-STEP*2./2.,-STEP*1./2.));'
        code += ' vec4 a13 = h13 * texture(nxtImg, uv2 + vec2(-STEP*2./2.,+STEP*1./2.));'
        code += ' vec4 a14 = h14 * texture(nxtImg, uv2 + vec2(-STEP*2./2.,+STEP*2./2.));'
        code += ' vec4 a21 = h21 * texture(nxtImg, uv2 + vec2(-STEP*1./2.,-STEP*2./2.));'
        code += ' vec4 a22 = h22 * texture(nxtImg, uv2 + vec2(-STEP*1./2.,-STEP*1./2.));'
        code += ' vec4 a23 = h23 * texture(nxtImg, uv2 + vec2(-STEP*1./2.,+STEP*1./2.));'
        code += ' vec4 a24 = h24 * texture(nxtImg, uv2 + vec2(-STEP*1./2.,+STEP*2./2.));'
        code += ' vec4 a31 = h31 * texture(nxtImg, uv2 + vec2(+STEP*1./2.,-STEP*2./2.));'
        code += ' vec4 a32 = h32 * texture(nxtImg, uv2 + vec2(+STEP*1./2.,-STEP*1./2.));'
        code += ' vec4 a33 = h33 * texture(nxtImg, uv2 + vec2(+STEP*1./2.,+STEP*1./2.));'
        code += ' vec4 a34 = h34 * texture(nxtImg, uv2 + vec2(+STEP*1./2.,+STEP*2./2.));'
        code += ' vec4 a41 = h41 * texture(nxtImg, uv2 + vec2(+STEP*2./2.,-STEP*2./2.));'
        code += ' vec4 a42 = h42 * texture(nxtImg, uv2 + vec2(+STEP*2./2.,-STEP*1./2.));'
        code += ' vec4 a43 = h43 * texture(nxtImg, uv2 + vec2(+STEP*2./2.,+STEP*1./2.));'
        code += ' vec4 a44 = h44 * texture(nxtImg, uv2 + vec2(+STEP*2./2.,+STEP*2./2.));'
        code += ' vec4 b = a11 + a12 + a13 + a14 + a21 + a22 + a23 + a24  +'
        code += '          a31 + a32 + a33 + a34 + a41 + a42 + a43 + a44;'
        #code += ' float wc = clamp(b.a/6.0 + a0.a, 0., 1.);' # WTF is this
        #code += ' float wc = a0.w;'
        code += ' float wc = a0.w;'
        code += ' vec3 a = (1.-wc) * b.rgb + a0.rgb;'
        #code += ' vec3 a = b.rgb;'
        #code += ' vec3 a = (1.-wc) * b.rgb;'
        #code += ' vec3 a =  texture(texSampler, uv).rgb;'
        #code += ' vec3 a =  a0.rgb;'
        #code += ' vec3 a = a0.rgb;'
        code += ' {} = vec4(a,1.);'.format(out)
        #code += ' {} = vec4(a0.rgb,1.);'.format(out)
        for i in range(1,5):
            for j in range(1,5): code = code.replace('h'+str(i)+str(j),str(h[i-1,j-1]))
        code = code.replace('STEP', step)
        return code

    push = '''
    #version 440
    in vec2 v_uv;
    layout(location = 0) out vec4 color;
    layout(binding = 0) uniform sampler2D lastImg;
    layout(location = 3) uniform float W;
    layout(location = 4) uniform int lvl;
    layout(location = 5) uniform int isLastLvl;
    void main() {
        int lvl1 = lvl-1;

        float P = (1. / (1<<(lvl1+1)));
        float uoff = .5 * (1 - pow(.25,(lvl1+1)/2)) / (1.-.25);
        float voff = .25 * (1 - pow(.25,(lvl1)/2)) / (1.-.25);
        vec2 uv = (4./3.) * (P * v_uv + vec2(uoff,voff));

        float step = (2./3.) / (W);
    ''' + conv_down(1, 'color', h1, 'step') + '''
    }'''
    pull = '''
    #version 440
    in vec2 v_uv;
    layout(location = 0) out vec4 color;
    layout(binding = 0) uniform sampler2D img;
    layout(binding = 1) uniform sampler2D nxtImg;
    layout(location = 3) uniform float W;
    layout(location = 4) uniform int lvl;
    void main() {

        int lvl0 = lvl+0;
        int lvl1 = lvl+1;
        float P0 = (1. / (1<<(lvl0+1)));
        float uoff0 = .5 * (1 - pow(.25,(lvl0+1)/2)) / (1.-.25);
        float voff0 = .25 * (1 - pow(.25,(lvl0)/2)) / (1.-.25);
        float P1 = (1. / (1<<(lvl1+1)));
        float uoff1 = .5 * (1 - pow(.25,(lvl1+1)/2)) / (1.-.25);
        float voff1 = .25 * (1 - pow(.25,(lvl1)/2)) / (1.-.25);

        vec2 uv = (4./3.) * (P0 * v_uv + vec2(uoff0,voff0));
        vec2 uv2 = (4./3.) * (P1 * v_uv + vec2(uoff1,voff1));
        float step = (2./3.) / W;

    ''' +conv_up(1, 'color', h2, 'step') + '''
    }
    '''

    push = push.replace('; ',';\n ')
    pull = pull.replace('; ',';\n ')
    pull = compileShader(pull, GL_FRAGMENT_SHADER)
    push = compileShader(push, GL_FRAGMENT_SHADER)
    push = compileProgram(vs,push)
    pull = compileProgram(vs,pull)

    return dict(pushShader=push,pullShader=pull)


class OcclusionAwareEngine:
    def __init__(self, meta):
        assert('w' in meta)
        meta.setdefault('occlusionLvls', 8)
        meta.setdefault('pullPushLvls', 8)
        for k,v in meta.items(): setattr(self,k,v)
        for k,v in makeShaders(self.occlusionLvls).items(): setattr(self,k,v)
        for k,v in makeShaders_pull_push(self.vs, self.pullPushLvls).items(): setattr(self,k,v)

        # Create Textures and FBOs
        self.sceneTexture, self.occlusionTexs = glGenTextures(1), glGenTextures(2)
        self.sceneDepthTex, self.pullPushTexs = glGenTextures(1), glGenTextures(2)

        self.pyrH = self.pyrW = (self.w*3)//2
        glBindTexture(GL_TEXTURE_2D, self.sceneTexture)
        #glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, self.w, self.h)
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, self.pyrW, self.pyrH)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, self.sceneDepthTex)
        #glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT24, self.w,self.h)
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT24, self.pyrW,self.pyrH)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        for t in (*self.pullPushTexs, *self.occlusionTexs):
            glBindTexture(GL_TEXTURE_2D, t)
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, self.pyrW, self.pyrH)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT)
        glBindTexture(GL_TEXTURE_2D, 0)


        self.fb0 = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fb0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  self.sceneDepthTex, 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+0, self.sceneTexture, 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+1, self.occlusionTexs[0], 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+2, self.occlusionTexs[1], 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+3, self.pullPushTexs[0], 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+4, self.pullPushTexs[1], 0)
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        z,o = -1,1
        self.pts = np.array((z,z,0,0,0, z,o,0,0,1, o,o,0,1,1, o,o,0,1,1, o,z,0,1,0, z,z,0,0,0),dtype=np.float32)

    def fullScreenQuad(self):
        glEnableVertexAttribArray(0); glEnableVertexAttribArray(1)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*5, self.pts)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*5, self.pts[3:])
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableVertexAttribArray(0); glDisableVertexAttribArray(1)

    def setRenderTarget(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fb0)

        glViewport(0,0,self.pyrW,self.pyrH)
        glDrawBuffers([GL_COLOR_ATTACHMENT0+i for i in range(1,5)])
        glClear(GL_COLOR_BUFFER_BIT)

        glViewport(0,0,self.w,self.h)
        glViewport(0,0,self.pyrW,self.pyrH)
        glDrawBuffer(GL_COLOR_ATTACHMENT0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0,0,self.w,self.h)

    def unsetRenderTarget(self):
        glActiveTexture(GL_TEXTURE0)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render(self, renderToScreen=True):
        glEnable(GL_TEXTURE_2D)
        glBindFramebuffer(GL_FRAMEBUFFER,self.fb0)

        proj = glGetFloatv(GL_PROJECTION_MATRIX)
        print('proj',proj)

        self.forward(proj, renderToScreen=renderToScreen)


    def forward(self, proj, renderToScreen):
        glEnable(GL_BLEND)
        #self.show('SCNEE', self.sceneTexture, time=1)
        #glBindTexture(GL_TEXTURE_2D, self.sceneDepthTex)
        #y = glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT)
        #print('ORIG DEPTH BOUNDS',y.min(), y.max())
        #cv2.imshow('orig depth', y); cv2.waitKey(1)

        #glViewport(0,0,self.pyrW,self.pyrH)
        #glDrawBuffers([GL_COLOR_ATTACHMENT0+i for i in range(1,5)])
        #glClear(GL_COLOR_BUFFER_BIT)

        glDisable(GL_DEPTH_TEST)

        # Unproject.
        glViewport(0,0,self.w,self.h)
        glUseProgram(self.unproject)
        inv_proj = np.copy(np.linalg.inv(proj).astype(np.float32), 'C')
        glUniformMatrix4fv(4, 1, False, inv_proj)
        #glUniformMatrix4fv(4, 1, True, inv_proj)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.sceneDepthTex)
        glDrawBuffer(GL_COLOR_ATTACHMENT1)
        #glClear(GL_COLOR_BUFFER_BIT)
        self.fullScreenQuad()

        #self.show('UNPROJECTED[0]', self.occlusionTexs[0], time=1)

        # Build XYZ pyramid.
        glUseProgram(self.pyrDownDepth)
        xoff,yoff,ww,hh = self.w,0,self.w//2,self.h//2
        glActiveTexture(GL_TEXTURE0)
        for i in range(1,self.occlusionLvls):
            glUniform1f(2, float(self.h))
            glUniform1i(3, i)

            glBindTexture(GL_TEXTURE_2D, self.occlusionTexs[1 - (i%2)])
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glDrawBuffer(GL_COLOR_ATTACHMENT1+(i%2))

            print(' - PyrDown:', i, xoff,yoff,ww,hh)
            glViewport(xoff,yoff,ww,hh)
            self.fullScreenQuad()
            if i % 2 == 0: xoff += ww
            else:          yoff += hh
            ww,hh = ww//2, hh//2

        #self.show('PYR[0]', self.occlusionTexs[0], time=1)
        #self.show('PYR[1]', self.occlusionTexs[1], time=1)

        # Get Mask. Directly write to first lvl of pullPush pyramid
        glUseProgram(self.getMask)
        for i in range(1,-1,-1):
            glActiveTexture(GL_TEXTURE0+i); glBindTexture(GL_TEXTURE_2D, self.occlusionTexs[i])
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, self.sceneTexture)
        glUniform1f(3, float(self.h))
        glViewport(0,0,self.w,self.h)

        # Draw directly to first PullPush tex
        glDrawBuffer(GL_COLOR_ATTACHMENT3)
        self.fullScreenQuad()

        #self.show('Mask[0]', self.pullPushTexs[0], time=1)

        # At this point, we will clear the occlusionTexs so we can re-use them
        glViewport(0,0,self.pyrW,self.pyrH)
        glDrawBuffers([GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2])
        glClear(GL_COLOR_BUFFER_BIT)

        # Do PullPush Algo to fill colors on non-occluded points.
        xoff,yoff,ww,hh = self.w,0,self.w//2,self.h//2
        glUseProgram(self.pushShader)
        glActiveTexture(GL_TEXTURE0)
        for i in range(1,self.pullPushLvls):
            glUniform1f(3, float(self.h))
            glUniform1i(4, i)
            glUniform1i(5, int(i == self.pullPushLvls-1))

            glBindTexture(GL_TEXTURE_2D, self.pullPushTexs[1 - (i%2)])
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

            glDrawBuffer(GL_COLOR_ATTACHMENT3+(i%2))

            glViewport(xoff,yoff,ww,hh)
            self.fullScreenQuad()
            if i < self.pullPushLvls - 1:
                if i % 2 == 0: xoff += ww
                else:          yoff += hh
                ww,hh = ww//2, hh//2

        #self.show('push[0]', self.pullPushTexs[0], time=1)
        #self.show('push[1]', self.pullPushTexs[1], time=1)

        glUseProgram(self.pullShader)
        for i in range(self.pullPushLvls-2,-1,-1):
            glUniform1f(3, float(self.h))
            glUniform1i(4, i)

            glActiveTexture(GL_TEXTURE1)
            #glBindTexture(GL_TEXTURE_2D, self.occlusionTexs[ (i%2)])
            glBindTexture(GL_TEXTURE_2D, self.occlusionTexs[1- (i%2)])
            #if i == self.pullPushTexs-2: glBindTexture(GL_TEXTURE_2D, self.occlusionTexs[1 - (i%2)])
            #else: glBindTexture(GL_TEXTURE_2D, self.pullPushTexs[1 - (i%2)])
            # This might help with black/white problem
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.pullPushTexs[(i%2)])
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

            #if i > 0: glDrawBuffer(GL_COLOR_ATTACHMENT1+1-(i%2))
            if renderToScreen:
                if i > 0: glDrawBuffer(GL_COLOR_ATTACHMENT1+(i%2))
                else: glBindFramebuffer(GL_FRAMEBUFFER, 0)
            else:
                glDrawBuffer(GL_COLOR_ATTACHMENT1+(i%2))
                glReadBuffer(GL_COLOR_ATTACHMENT1+(i%2))
            '''
            glDrawBuffer(GL_COLOR_ATTACHMENT1+(i%2))
            glReadBuffer(GL_COLOR_ATTACHMENT1+(i%2))
            '''

            ww,hh = ww*2, hh*2
            if i % 2 == 0: xoff -= ww
            else:          yoff -= hh
            print(' - Pull', i, xoff,yoff,ww,hh)
            glViewport(xoff,yoff,ww,hh)
            self.fullScreenQuad()


        #self.show('pulled[0]', self.occlusionTexs[0], time=1)
        #self.show('pulled[1]', self.occlusionTexs[1], time=1)

        '''
        if renderToScreen:
            glUseProgram(self.simpleTextured)
            glActiveTexture(GL_TEXTURE0)
            #glBindTexture(GL_TEXTURE_2D, self.sceneDepthTex)
            #glBindTexture(GL_TEXTURE_2D, self.pullPushTexs[0])
            glBindTexture(GL_TEXTURE_2D, self.occlusionTexs[0])
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0,0,self.w,self.h)
            glClearColor(1,0,1,1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.fullScreenQuad()
        '''

        glUseProgram(0)
        '''
        if renderToScreen:
            #glUseProgram(self.simpleTextured)
            #glActiveTexture(GL_TEXTURE0)
            #glBindTexture(GL_TEXTURE_2D, self.sceneDepthTex)
            #glBindTexture(GL_TEXTURE_2D, self.pullPushTexs[0])
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0,0,self.w,self.h)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.fullScreenQuad()
        else:
            glDrawBuffer(GL_COLOR_ATTACHMENT3)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.fullScreenQuad()
        '''


    def show(self, name, buf=None,time=1):
        glBindTexture(GL_TEXTURE_2D, buf)

        y = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
        #y = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
        #y = np.frombuffer(y,dtype=np.uint8).reshape(self.h>>i, self.w>>i, -1)
        #y = np.frombuffer(y,dtype=np.uint8).reshape(self.h, self.w, -1)
        glBindTexture(GL_TEXTURE_2D, 0)
        y_ = y[...,:3]
        print(y_, y_.reshape(-1,3).min(0), y_.reshape(-1,3).max(0))
        #y_ = y[...,2]; y_ = (y_ - y_.min()); y_ = y_ / y_.max()
        #y_[...,2] = (y_[...,2] - y_[...,2].min()); y_[...,2] = y_[...,2] / y_[...,2].max()
        #print(y_)
        #print(y_, y_.reshape(-1,3).min(0), y_.reshape(-1,3).max(0))

        #y_ = cv2.pyrDown(y_)
        #y_ = cv2.pyrDown(y_)
        #y_ = y_[::-1]

        cv2.imshow(name+'_y',y_)
        #cv2.imshow(name+'_yweight',y[...,3])
        if time > 0: cv2.waitKey(time)
