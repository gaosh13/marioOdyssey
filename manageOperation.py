import sys
import pathlib

if len(sys.argv) < 2:
	print("Usage: [initial] [delete number]")

p = pathlib.Path('./operation')

if sys.argv[1] == "initial":
	for i in range(9):
		q = p / str(i)
		q.makedirs(parents=True, exist_ok=True)

if sys.argv[1] == "delete":
	if len(sys.argv) == 3:
		num = sys.argv[2]
		for i in range(9):
			q = p / str(i)
			gl = q.glob("frame_%s_*.jpg" % num) if num != -1 else q.glob("*")
			for j in gl:
				j.unlink()
