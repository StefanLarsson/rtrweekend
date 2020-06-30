
def make_test_ppm(w, h):
	print ("P3")
	print (str(w) + " " + str(h))
	print (255)
	
	for j in range(h - 1, 0, -1):
		for i in range(w):
			r = i / (w - 1.0)
			g = j / (h - 1.0)
			b = 0.25
			ri = int( r * 255)
			gi = int( g * 255)
			bi = int( b * 255)
			print (ri, gi, bi)
			



make_test_ppm(256, 256)

