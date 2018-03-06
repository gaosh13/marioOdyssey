import serial

class Serial:
    def __init__(self, BAUD=57600, PORT="COM4"):
        self.ser = serial.Serial()
        self.ser.baudrate = BAUD
        self.ser.port = PORT
        self.ser.open()
        self.operation = ""
    
    def qsend(self, X, Y):
        if X != "" or Y != "":
            self.ser.write(bytes(X + Y, 'ascii'))
            # print(X + Y)

    def send(self):
        self.ser.write(bytes(self.operation, 'ascii'))

    def operate(self, X, Y):
        self.operation = X + " " + Y
    
    def close(self):
        self.ser.close()

class Serial_null:
    def __init__(self, BAUD=57600, PORT="COM4"):
        print("Serial test")
    
    def qsend(self, X, Y):
        pass
    
    def send(self):
        pass
    
    def operate(self, X, Y):
        pass
    
    def close(self):
        pass