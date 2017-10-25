import sys
import numpy as np
import transformations as tr

def get_cam():
	data = {}
	ridx = 0
	kfd = 0
	rotm2 = []
	for line in open('result.txt').readlines():
		if line[:4] != 'cam ':
			f = line.split()
			if len(f) != 4: print f
			ridx += 1
			rotm2.append(np.array(f).astype(np.float))
			if ridx == 4:
				rotm = conv2rotm(data[kfd])
				if not np.allclose(rotm, rotm2, 0.01):
					print 'not match',kfd, data[kfd]
					print rotm
					print rotm2
				ridx = 0
				rotm2 = []
			continue
		line = line[4:]
		i,t,x,y,z,qx,qy,qz,qw = line.split()
		v = (x,y,z,qx,qy,qz,qw)
		v = np.array(v).astype(np.float)
		data[i] = v
		kfd = i
		x,y,z,w = v[3:7]
		# should be 1, print x*x+y*y+z*z+w*w
	return data

def get_point_cloud():
  data = {}
  status = ''
  last = ''
  for line in open("log").readlines():
    if line[:4] == 'kfd ':
        f = line.split()
        status = f[1]
        if status not in data: data[status] = []
        if status != last: data[status] = []
        last = status
        data[f[1]].append([f[2],f[3],f[4]])
        continue
    elif line.find('end-of-') > 0:
        status = ''
        continue
    if status == '': continue
    i,x,y,z = line.split()
    if 'nan' in [x,y,z]: continue
    data[status].append([x,y,z])
  r = {}
  for k in (data):
      v = data[k]
      r[k] = np.array(v).astype(np.float)
  return r

#print(a.shape)
#d = np.amin(a, axis=0)
#a -= d
#mx,my,mz = (np.amax(a, axis=0))

def gen_js(a):
  print 'var a = [ ',
  for x,y,z in a:
    print [x,y,z],
    print(',')

  print '[0,0,0]]'

def gen_pcd(a):
	n = len(a)
	print '''VERSION .7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH %d
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS %d''' % (n,n)
	print 'DATA ascii'
	s = 25
	for x,y,z in a:
		print  x/s,y/s,z/s

def conv2rotm(v):
	x,y,z,i,j,k,r = v
	#m = tr.rotation_matrix(qw*np.pi/180,(qx,-qy,qz))
	m=np.zeros((4,4))
	m[3,3] = 1
	m[0,0] = 1-2*(j*j+k*k)
	m[1,1] = 1-2*(i*i+k*k)
	m[2,2] = 1-2*(j*j+i*i)
	m[0,1] = 2*(i*j-k*r)
	m[1,2] = 2*(k*j-i*r)
	m[2,0] = 2*(i*k-j*r)
	m[0,2] = 2*(i*k+j*r)
	m[1,0] = 2*(i*j+k*r)
	m[2,1] = 2*(j*k+i*r)
	m[:3,3] = (x,y,z)
	return m

data = []
cam = get_cam()
pc = get_point_cloud()
for i in pc:
	if i not in cam: continue
	r = conv2rotm(cam[i])
	for p in pc[i]:
		p = np.append(p,[1])
		data.append(r.dot(p)[:3])

#gen_pcd(data)
gen_js(data)

cam2 = {}
for k in cam:
	cam2[int(k)] = cam[k]
a = []
for k in sorted(cam2.iterkeys()):
	x,y,z,i,b,c,d = cam2[k]
	a.append([x,y,z])
#gen_js(a)
