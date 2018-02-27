from pynput.keyboard import Key, Listener
from string import ascii_lowercase
from VolleyBall import Volleyball 
import threading
keys_pressed = ""

def on_press(key):
	# print("{0} pressed".format(key))
	global keys_pressed
	if key == Key.esc:
		return False
	try:
		c = key.char
		if c not in keys_pressed:
			keys_pressed += c
	except Exception as e:
	 	print(e)

def on_release(key):
	# print("{0} released".format(key))
	global keys_pressed
	if key == Key.esc:
		return False
	try:
		c = key.char
		if c in keys_pressed:
			keys_pressed = keys_pressed.replace(c, "")
	except Exception as e:
		print(e)

def keyboard_listener():
	with Listener(on_press=on_press,
		on_release=on_release) as listener:
		listener.join()	

def keys_to_action():
	if len(keys_pressed) == 0:
		return 0
	if keys_pressed[0] == 'w':
		return 1
	elif keys_pressed[0] == 'a':
		return 2
	elif keys_pressed[0] == 's':
		return 3
	elif keys_pressed[0] == 'd':
		return 4

def main():
	threads = []
	game = Volleyball()
	t = threading.Thread(target=keyboard_listener)
	threads.append(t)
	t.start()
	while True:
		if not game.step(keys_to_action())[0]:
			break

if __name__ == '__main__':
	main()