import sys
import math
import numpy as np


class ray:
	"""A ray class"""
	def __init__(self, origin, direction):
		self.origin = origin
		self.direction = direction

	def at(self, t):
		return self.origin + t * self.direction


class sphere:
	""""A sphere class"""
	def __init__(self, centre, radius):
		self.centre = centre
		self.radius = radius

	
def unit_vector(v):
	n = np.linalg.norm(v)
	return v * ( 1/ n)
		
def hit_sphere(center, radius, ray):
	oc = ray.origin - center
	a = np.dot(ray.direction, ray.direction)
	b = 2.0 * np.dot (oc, ray.direction)
	c = np.dot(oc, oc) - radius * radius
	disc = b * b - 4 * a * c
	return disc > 0

def hit_sphere2(center, radius, ray):
	oc = ray.origin - center
	a = np.dot(ray.direction, ray.direction)
	half_b = np.dot (oc, ray.direction)
	c = np.dot(oc, oc) - radius * radius
	disc = half_b * half_b - a * c
	if disc < 0:
		return -1
	else:
		return (- half_b - math.sqrt(disc)) / a
class hit_record:
	def __init__(self, t, p, normal):
		self.t = t
		self.p = p
		self.normal = normal

def hit_sphere3(center, radius, ray):
	oc = ray.origin - center
	a = np.dot(ray.direction, ray.direction)
	half_b = np.dot (oc, ray.direction)
	c = np.dot(oc, oc) - radius * radius
	disc = half_b * half_b - a * c
	if disc < 0:
		return None
	else:
		root = math.sqrt(disc)
		temp = (- half_b - root) / a
		if (temp > 0):
			p = ray.at(temp)
			normal = (p - center) / radius
			if np.dot(normal, ray.direction) > 0:
				normal = -  normal
			result = hit_record(temp, p, normal)
			return result
		temp = (- half_b + root) / a
		if (temp > 0):
			p = ray.at(temp)
			normal = (p - center) / radius
			if np.dot(normal, ray.direction) > 0:
				normal = -  normal
			result = hit_record(temp, p, normal)
			return result
	return None
	
def ray_color(r):
	t = hit_sphere2(np.array([0,0,-1]), 0.5, r);
	if ( t > 0.0):
		N = unit_vector(r.at(t) - np.array([0,0,-1]))
		return 0.5 * ( N + 1.0)
	unit = unit_vector(r.direction)
	t = 0.5 * (unit[1] + 1.0)
	return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + np.array([0.5, 0.7, 1.0])

def hit(r, world):
	closest = 100000000000
	res = None
	for w in world:
		ahit = hit_sphere3(w.centre, w.radius, r)
		if ahit and ahit.t < closest:
			#print ("Ray {0} changihng to sphere {1}".format(r,w))
			closest = ahit.t
			res = ahit

	return res

def ray_color(r, world):
	ahit = hit(r, world)
	if ahit:
		return 0.5 * (ahit.normal + 1.0)

	unit = unit_vector(r.direction)
	t = 0.5 * (unit[1] + 1.0)
	return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + np.array([0.5, 0.7, 1.0])

def write_color(color, stream):
	r = int(color[0] * 256 )
	g = int(color[1] * 256 )
	b = int(color[2] * 256 )
	
	stream.write("{0} {1} {2}\n".format(r,g,b))
	
def make_ray_ppm(stream = sys.stdout):
	aspect_ratio = 16  / 9
	width = 384
	height = int (width / aspect_ratio)

	world = list()
	world.append(sphere(np.array([0,0,-1]), 0.5))
	world.append(sphere(np.array([0,-100.50,-1]), 100))

	vp_height = 2.0
	vp_width = aspect_ratio * vp_height

	focal_length = 1.0
	origin = np.array([0,0,0])
	horiz = np.array([vp_width, 0, 0])
	vert  = np.array([0, vp_height, 0])
	lower_left = origin - (horiz / 2) - (vert / 2) - np.array([0,0,focal_length])
	stream.write("P3\n{0} {1}\n255\n".format(width, height))
	for j in range(height - 1, 0, -1):
		for i in range(width):
			u = i / (width - 1)
			v = j / (height - 1)
			r = ray(origin, lower_left + u * horiz + v * vert - origin)
			color = ray_color(r, world)
			write_color(color, stream)

	
	
	
	
def make_test_ppm(w, h, stream = sys.stdout):


	stream.write ("P3\n")
	stream.write(str(w) + " " + str(h) + '\n')
	stream.write ("{0}\n".format(255))
	
	for j in range(h - 1, 0, -1):
		sys.err.write("Line {0}".format(j))
		for i in range(w):
			r = i / (w - 1.0)
			g = j / (h - 1.0)
			b = 0.25
			ri = int( r * 255)
			gi = int( g * 255)
			bi = int( b * 255)
			stream.write ("{0} {1} {2}\n".format(ri, gi, bi))
			



# make_test_ppm(256, 256)
make_ray_ppm()

