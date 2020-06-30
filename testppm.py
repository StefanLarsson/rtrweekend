import sys
def make_test_ppm(w, h, stream = sys.stdout):


	stream.write ("P3\n")
	stream.write(str(w) + " " + str(h) + '\n')
	stream.write ("{0}\n".format(255))
	
	for j in range(h - 1, 0, -1):
		for i in range(w):
			r = i / (w - 1.0)
			g = j / (h - 1.0)
			b = 0.25
			ri = int( r * 255)
			gi = int( g * 255)
			bi = int( b * 255)
			stream.write ("{0} {1} {2}\n".format(ri, gi, bi))
			



make_test_ppm(256, 256)

