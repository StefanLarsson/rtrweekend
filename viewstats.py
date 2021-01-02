import pstats
p = pstats.Stats('testit')
p.sort_stats('tottime')
p.print_stats()
p.print_callees()
