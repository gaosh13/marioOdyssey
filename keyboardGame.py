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
	if all(key in keys_pressed for key in 'wa'):
		return 5
	elif all(key in keys_pressed for key in 'sa'):
		return 6
	elif all(key in keys_pressed for key in 'sd'):
		return 7
	elif all(key in keys_pressed for key in 'wd'):
		return 8
	elif 'w' in keys_pressed:
		return 1
	elif 'a' in keys_pressed:
		return 2
	elif 's' in keys_pressed:
		return 3
	elif 'd' in keys_pressed:
		return 4
	return 0

def main():
	threads = []
	game = Volleyball(collection=True)
	t = threading.Thread(target=keyboard_listener)
	threads.append(t)
	t.start()
	while t.is_alive():
		if not game.step(keys_to_action())[0]:
			break
	game.close()

if __name__ == '__main__':
	main()