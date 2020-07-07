import sys
import math
import random
import numpy as np
import time
import profile


class ray:
	"""A ray class"""
	def __init__(self, origin, direction):
		self.origin = origin
		self.direction = direction

	def at(self, t):
		return self.origin + t * self.direction


class sphere:
	"""A sphere class"""
	def __init__(self, centre, radius, scatter):
		self.centre = centre
		self.radius = radius
		self.scatter = scatter
	def hit(self, ray, tmin, tmax):
		return hit_sphere(self.centre, self.radius, ray, tmin, tmax, self.scatter)


class camera:
	"""A camera"""
	def __init__(self):
		self.aspect = 16 / 9
		self.vp_h = 2.0
		self.vp_w = self.aspect * self.vp_h
		self.focal = 1.0

		self.origin = np.array([0.0,0.0,0.0])
#		self.origin = np.array([0.0,0.0,1.0])
		self.horiz =  np.array([self.vp_w, 0,0])
		self.vert =  np.array([0,self.vp_h,0])
		self.lower_left = self.origin - self.horiz / 2 - self.vert / 2 - np.array([0,0,self.focal])

	def get_ray(self, u,v):
		return ray(self.origin, self.lower_left + u * self.horiz + v * self.vert - self.origin)


def random_unit_helper():
	return 2 * np.random.random_sample(3) - 1

def random_in_unit():
	while True:
		v = random_unit_helper()
		if np.dot(v,v) <= 1.0:
			return v
def unit_vector(v):
	n = np.linalg.norm(v)
	return v * ( 1/ n)

class hit_record:
	def __init__(self, t, p, normal, scatter):
		self.t = t
		self.p = p
		self.normal = normal
		self.scatter = scatter

def hit_sphere(center, radius, ray, tmin, tmax, scatter):
	dot = np.dot
	oc = ray.origin - center
	a = dot(ray.direction, ray.direction)
	half_b = dot (oc, ray.direction)
	c = dot(oc, oc) - radius * radius
	disc = half_b * half_b - a * c
	if disc < 0:
		return None
	else:
		root = math.sqrt(disc)
		temp = (- half_b - root) / a
		if (temp > tmin  and temp < tmax):
			p = ray.at(temp)
			normal = (p - center) / radius
			if dot(normal, ray.direction) > 0:
				normal = -  normal
			result = hit_record(temp, p, normal, scatter)
			return result
		temp = (- half_b + root) / a
		if (temp > tmin and temp < tmax):
			p = ray.at(temp)
			normal = (p - center) / radius
			if dot(normal, ray.direction) > 0:
				normal = -  normal
			result = hit_record(temp, p, normal, scatter)
			return result
	return None


c1g = np.array([1.0, 1.0, 1.0])
c2g = np.array([0.5, 0.7, 1.0])

def hit(r, world, tmin , tmax):
	closest = tmax
	res = None
	for w in world:
		ahit = w.hit(r, tmin, tmax)
		if ahit and ahit.t < closest:
			closest = ahit.t
			res = ahit

	return res


class scattered:
	def __init__(self, direction, attenuation):
		self.direction = direction
		self.attenuation = attenuation


class lambertian:
	def __init__(self, color):
		self.albedo = color

	def scatter(self, incoming, hit):
		attenuation = self.albedo
		scattered_direction = hit.normal + random_in_unit()
		scattered_ray = ray(hit.p, scattered_direction)
		return scattered(scattered_ray, attenuation)

def reflect(v, n):
	return v - 2 * np.dot(v,n) * n

class metal:
	def __init__(self, color):
		self.albedo = color
	def scatter(self, incoming,hit):
		reflected = reflect(incoming.direction, hit.normal)
		if np.dot(reflected, hit.normal) <= 0:
			return None
		attenuation = self.albedo
		scattered_ray = ray(hit.p, reflected)
		return scattered(scattered_ray, attenuation)

def ray_color(r, world, depth):
	if (depth <= 0):
		return np.array([0.0, 0.0, 0.0])
	ahit = hit(r, world, 0, float("inf"))
	if ahit:
		scattered = ahit.scatter.scatter(r,ahit)
		if scattered:
			attenuation = scattered.attenuation
			direction = scattered.direction
			return attenuation * ray_color(direction, world, depth - 1)
		return np.array([0.0,0.0,0.0])
#		dir = ahit.normal + random_in_unit()
#		return 0.5 * ray_color(ray(ahit.p, dir), world, depth - 1)

	unit = unit_vector(r.direction)
	t = 0.5 * (unit[1] + 1.0)
	return (1.0 - t) * c1g + c2g

def write_color(color, stream):

	r = int(math.sqrt(color[0]) * 255)
	g = int(math.sqrt(color[1]) * 255)
	b = int(math.sqrt(color[2]) * 255)
	stream.write("{0} {1} {2}\n".format(r,g,b))

def make_ray_ppm(stream = sys.stdout):
	aspect_ratio = 16  / 9
	width = 384
	height = int (width / aspect_ratio)
	samples_per_pixel = 100
#	samples_per_pixel = 50
#	samples_per_pixel = 5
	max_depth = 50

	world = list()

	material = lambertian(np.array([0.7, 0.3, 0.3]))
	world.append(sphere(np.array([0,0,-1]), 0.5, material))

	material2 = lambertian(np.array([0.8, 0.8, 0.0]))
	world.append(sphere(np.array([0,-100.5,-1]), 100, material2))

	material3 = metal(np.array([0.8, 0.6, 0.2]))
	world.append(sphere(np.array([1.0,0.0,-1.0]), 0.5, material3))

	material4 = metal(np.array([0.8, 0.8, 0.8]))
	world.append(sphere(np.array([-1.0,0.0,-1.0]), 0.5, material4))

	cam = camera()
	stream.write("P3\n{0} {1}\n255\n".format(width, height))
	for j in range(height - 1, 0, -1):
		sys.stderr.write("{0} rows to go\n".format(j))
		for i in range(width):
			color = np.array([0.0,0.0,0.0])
			for k in range(samples_per_pixel):
				u = (i + random.random()) / (width - 1)
				v = (j + random.random()) / (height - 1)
				r = cam.get_ray(u,v)
				color += ray_color(r, world, max_depth)
			color /=   samples_per_pixel
			write_color(color, stream)
def main():
	#profile.run('make_ray_ppm()')
	make_ray_ppm()

if __name__ == "__main__":
	main()

