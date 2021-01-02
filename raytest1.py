import sys
import math
import random
import numpy as np
import time
import cProfile as profile


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
    def __init__(self, lookfrom, lookat, vup, vfov, aspect):
        aspect = 16.0 / 9
        h = math.tan(vfov / 2)
        vp_h = 2.0 * h
        vp_w = aspect * vp_h
        focal = 1.0
        
        w = unit_vector (lookfrom - lookat)
        u = unit_vector (cross(vup, w))
        v = cross(w,u)


        self.origin = lookfrom
        self.horiz =  vp_w * u
        self.vert =  vp_h *v
        self.lower_left = self.origin - self.horiz / 2 - self.vert / 2 - w
        self.helper_offset = self.lower_left - self.origin

    def get_ray(self, u,v):
        w = self.helper_offset + u * self.horiz + v * self.vert
        w = w * (1.0 / math.sqrt(np.dot(w,w)))
        #return ray(self.origin, self.helper_offset + u * self.horiz + v * self.vert)
        return ray(self.origin, w)


def random_unit_helper():
    return 2.0 * np.random.random_sample(3) - 1.0

def random_in_unit():
    while True:
        v = random_unit_helper()
        if np.dot(v,v) <= 1.0:
            return v
def unit_vector(v):
    n = math.sqrt(np.dot(v,v))
    return v * ( 1/ n)


def cross(u, v):
    """Cross product of 3d vectors"""
    return np.array([u[1]*v[2] - u[2] * v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]])

class hit_record:
    def __init__(self, t, p, normal, scatter):
        self.t = t
        self.p = p
        self.normal = normal
        self.scatter = scatter

    def set_face_normal(self, ray, outward_normal):
        self.front_face = np.dot(ray.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal

def hit_sphere(center, radius, ray, tmin, tmax, scatter):
    dot = np.dot
    oc = ray.origin - center
    #a = dot(ray.direction, ray.direction) #independent of the sphere!
    #print ("a = {}".format(dot(ray.direction, ray.direction)))
    a = 1.0
    half_b = dot (oc, ray.direction) # = dot(ray.origin - center, ray.direction)  = dot(ray.origin, ray.direction) - dot (center, ray.direction)
    c = dot(oc, oc) - radius * radius
    disc = half_b * half_b - a * c
    if disc < 0.0:
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
            result.set_face_normal(ray, (p - center) / radius)
            return result
        temp = (- half_b + root) / a
        if (temp > tmin and temp < tmax):
            p = ray.at(temp)
            normal = (p - center) / radius
            if dot(normal, ray.direction) > 0:
                normal = -  normal
            result = hit_record(temp, p, normal, scatter)
            result.set_face_normal(ray, (p - center)/ radius)
            return result
    return None

def hit(r, world, tmin , tmax):
    closest = tmax
    res = None
    for w in world:
        ahit = w.hit(r, tmin, tmax) # this is creating new hit records all the time?
        if ahit and ahit.t < closest:
            closest = ahit.t
            res = ahit

    return res


class scattered:
    def __init__(self, direction, attenuation):
        self.direction = direction
        self.direction.direction = direction.direction * (1.0 / (math.sqrt(np.dot(direction.direction, direction.direction))))
        self.attenuation = attenuation


class lambertian:
    def __init__(self, color):
        self.albedo = color

    def scatter(self, incoming, hit):
        attenuation = self.albedo
        #scattered_direction = hit.normal
        scattered_direction = hit.normal + random_in_unit()
        scattered_ray = ray(hit.p, scattered_direction)
        return scattered(scattered_ray, attenuation)

def reflect(v, n):
    """Reflect the vector v using the unit vector n"""
    return v - 2 * np.dot(v,n) * n

class metal:
    def __init__(self, color, fuzz):
        self.albedo = color
        self.fuzz = fuzz
    def scatter(self, incoming,hit):
        reflected = reflect(incoming.direction, hit.normal)
        reflected = reflected + self.fuzz * random_in_unit()
        if np.dot(reflected, hit.normal) <= 0:
            return None
        attenuation = self.albedo
        scattered_ray = ray(hit.p, reflected)
        return scattered(scattered_ray, attenuation)


def refract (u, n, relative_index):
    cos_theta = np.dot(-u,n)
    out_perp = relative_index * (u + cos_theta * n)
    out_para = -math.sqrt(1.0 - np.dot(out_perp, out_perp)) * n
    return out_perp + out_para


def reflectance(cosine, relative_index):
    """Schlick approximation"""
    r0 = ( 1.0 - relative_index)  / ( 1.0 + relative_index)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * math.pow( 1.0 - cosine, 5)

class dielectric:
    def __init__(self, index):
        self.index = index

    def scatter(self, incoming, hit):
        attenuation = np.array([1.0, 1.0, 1.0])
        relative_index = 1.0 / self.index if hit.front_face else self.index
        unit = unit_vector(incoming.direction)
        cos_theta = -np.dot(unit, hit.normal)
        r2 = relative_index * relative_index
        if r2 * cos_theta * cos_theta < (r2 - 1.0) or reflectance(cos_theta, relative_index) > np.random.random():
            # total internal reflection or just reflection anyway
            direction = reflect(unit, hit.normal)
        else:
            direction = refract(unit, hit.normal, relative_index)
        return scattered(ray(hit.p, direction), attenuation)

c1g = np.array([1.0, 1.0, 1.0])
c2g = np.array([0.5, 0.7, 1.0])

def ray_color(r, world, depth):
    if (depth <= 0):
        return np.array([0.0, 0.0, 0.0])
    ahit = hit(r, world, 0.001, float("inf"))
    if ahit:
        scattered = ahit.scatter.scatter(r,ahit)
        if scattered:
            attenuation = scattered.attenuation
            direction = scattered.direction
            return attenuation * ray_color(direction, world, depth - 1)
        return np.array([0.0,0.0,0.0])
    # if we pass here we have hit nothing and will use the sky colour
    unit = unit_vector(r.direction)
    t = 0.5 * (unit[1] + 1.0)
    return (1.0 - t) * c1g + c2g

def write_color(color, stream):

    r = int(math.sqrt(color[0]) * 255)
    if r > 255:
        r = 255
    g = int(math.sqrt(color[1]) * 255)
    if g > 255:
        g = 255
    b = int(math.sqrt(color[2]) * 255)
    if b > 255:
        b = 255
    stream.write("{0} {1} {2}\n".format(r,g,b))

def three_spheres_world():
    world = list()

    material_ground = lambertian(np.array([0.8, 0.8, 0.0]))
    #material_center = lambertian(np.array([0.7, 0.3, 0.3]))
    material_center = lambertian(np.array([0.1, 0.2, 0.5]))
    # material_left = metal(np.array([0.8, 0.8, 0.8]), 0.3)
    material_left = dielectric(1.5)
    material_right = metal(np.array([0.8, 0.6, 0.2]), 0.0)

    world.append(sphere(np.array([0.0,-100.5,-1.0]), 100.0, material_ground))
    world.append(sphere(np.array([0.0,0.0,-1.0]), 0.5, material_center))
    world.append(sphere(np.array([-1.0,0.0,-1.0]), 0.5, material_left))
    world.append(sphere(np.array([-1.0,0.0,-1.0]), -0.4, material_left))
    world.append(sphere(np.array([1.0,0.0,-1.0]), 0.5, material_right))
    return world

def random_world():
    world = list()

    material_ground = lambertian(np.array([0.5, 0.5, 0.5]))
    world.append(sphere(np.array([0.0,-1000.0,0.0]), 1000.0, material_ground))

    material1 = dielectric(1.5);
    world.append(sphere(np.array([-0.0, 1.0, 0.0]), 1.0, material1));

    material2 = lambertian(np.array([0.4, 0.2, 0.1]));
    world.append(sphere(np.array([-4.0, 1.0, 0.0]), 1.0, material2));

    material3 = metal(np.array([0.7, 0.6, 0.5]), 0.0);
    world.append(sphere(np.array([4.0, 1.0, 0.0]), 1.0, material3));

    for a in range (-5, 5):
        for b in range (-5, 5):
            centre = np.array([a + random.random(), 0.2, b + random.random()])
            material = lambertian(np.random.random_sample(3))
            world.append(sphere(centre, 0.2, material))
    return world


def make_ray_ppm(stream = sys.stdout):
    aspect_ratio = 16.0  / 9
    width = 400
    #width = 100
    height = int (width / aspect_ratio)
#    samples_per_pixel = 100
#    samples_per_pixel = 50
    samples_per_pixel = 10
#    samples_per_pixel = 1
    max_depth = 50

    #world = three_spheres_world()
    world = random_world()

    #cam = camera(np.array([-2.0, 2.0, 1.0]), np.array([0.0,0.0,-1.0]),np.array([0.0, 1.0, 0.0]), math.pi/9.0 , aspect_ratio )
    cam = camera(np.array([13.0, 2.0, 3.0]), np.array([0.0,0.0,-0.0]),np.array([0.0, 1.0, 0.0]), math.pi/9.0 , aspect_ratio )
    stream.write("P3\n{0} {1}\n255\n".format(width, height))
    for j in range(height - 1, -1, -1):
        sys.stderr.write("{0} rows to go\n".format(j))
        for i in range(width):
            color = np.array([0.0,0.0,0.0])
            for k in range(samples_per_pixel):
#                u = (i + random.random()) / (width - 1)
#                v = (j + random.random()) / (height - 1)
                u = (i + random.random()) / (width)
                v = (j + random.random()) / (height)
#                u = (i + 0.5) / (width - 1)
#                v = (j + 0.5) / (height - 1)
                r = cam.get_ray(u,v)
                color += ray_color(r, world, max_depth)
            color /=   samples_per_pixel
            write_color(color, stream)
def main():
    profile.run('make_ray_ppm()', 'testit')
#    profile.run('make_ray_ppm()')
    #make_ray_ppm()

if __name__ == "__main__":
    main()

